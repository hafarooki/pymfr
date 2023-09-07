import scipy.integrate
import scipy.constants
from scipy.signal import savgol_coeffs
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity


class ReconstructedMap():
    def __init__(self, error_fit,
                 magnetic_potential,
                 magnetic_field_x,
                 magnetic_field_y,
                 magnetic_field_z,
                 current_density_x,
                 current_density_y,
                 current_density_z,
                 gas_pressure,
                 core_mask) -> None:
        self.error_fit = error_fit
        self.magnetic_potential = magnetic_potential
        self.magnetic_field_x = magnetic_field_x
        self.magnetic_field_y = magnetic_field_y
        self.magnetic_field_z = magnetic_field_z
        self.current_density_x = current_density_x
        self.current_density_y = current_density_y
        self.current_density_z = current_density_z
        self.gas_pressure = gas_pressure
        self.core_mask = core_mask



def reconstruct_map(magnetic_field_observed, gas_pressure_observed, sample_spacing, alpha=None,
                              resolution = 21, poly_order: int = 3, aspect_ratio: float = 1):
    magnetic_field_observed = torch.as_tensor(magnetic_field_observed)
    dtype = magnetic_field_observed.dtype
    device = magnetic_field_observed.device
    gas_pressure_observed = torch.as_tensor(gas_pressure_observed, dtype=dtype, device=device)
    if alpha is None:
        alpha = torch.zeros(gas_pressure_observed.shape[:-1], device=device, dtype=dtype)
    alpha = torch.as_tensor(alpha, dtype=dtype, device=device).reshape(gas_pressure_observed.shape[:-1] + (1,))
    sample_spacing = torch.as_tensor(sample_spacing, dtype=dtype, device=device)

    return _reconstruct_map(magnetic_field_observed,
                                            gas_pressure_observed,
                                            sample_spacing,
                                            alpha,
                                            resolution,
                                            poly_order,
                                            aspect_ratio)

#@torch.jit.script
def _laplacian(potential, potential_peak, poly_order: int, field_line_quantity_coeffs):
    tail_coefficient_left = field_line_quantity_coeffs[..., 0]
    tail_exponent_left = field_line_quantity_coeffs[..., 1] / tail_coefficient_left
    peak_sign = torch.sign(potential_peak)

    return torch.where(torch.sign(potential) == peak_sign.unsqueeze(-1),
                                     (torch.stack([i * potential ** (i - 1) for i in range(1, poly_order + 1)], dim=-1) @ field_line_quantity_coeffs[..., 1:].unsqueeze(-1)).squeeze(-1),
                                     (tail_coefficient_left * tail_exponent_left).unsqueeze(-1) * torch.exp(tail_exponent_left.unsqueeze(-1) * potential))
    

#@torch.jit.script
def _polyfit(potential, quantity, polyorder: int):
    x = torch.stack([potential ** i for i in range(polyorder + 1)], dim=-1) 

    coeffs = torch.linalg.lstsq(x, quantity).solution
    return coeffs


#@torch.jit.script
def _evaluate_fit_values(coeffs, potential, peak_sign):
    tail_coefficient_left = coeffs[..., 0]
    tail_exponent_left = coeffs[..., 1] / tail_coefficient_left 

    return torch.where(torch.sign(potential) == peak_sign.reshape(peak_sign.shape + (1,) * (len(potential.shape) - len(peak_sign.shape))),
                                    (torch.stack([potential ** i for i in range(coeffs.shape[-1])], dim=-1) @ coeffs.unsqueeze(-1)).squeeze(-1),
                                    tail_coefficient_left.unsqueeze(-1) * torch.exp(tail_exponent_left.unsqueeze(-1) * potential))


#@torch.jit.script
def _three_point_smooth(solution, height: float):
    # note: this is an in-place operation
    center_weight = 1 - torch.abs(height) / 3
    edge_weight = (1 - center_weight) / 2
    solution[..., 1:-1] = center_weight * solution[..., 1:-1] + edge_weight * (solution[..., :-2] + solution[..., 2:])
    return solution


#@torch.jit.script
def _second_derivative(solution, dx: int = 1):
    result = torch.empty_like(solution)
    result[..., 1:-1] = solution[..., 2:] - 2 * solution[..., 1:-1] + solution[..., :-2]
    result[..., 0] = 2 * solution[..., 0] - 5 * solution[..., 1] + 4 * solution[..., 2] - solution[..., 3]
    result[..., -1] = 2 * solution[..., -1] - 5 * solution[..., -1 - 1] + 4 * solution[..., -1 - 2] - solution[..., -1 - 3]
    return result / dx ** 2


#@torch.jit.script
def _generate_connection_map(potential):
    # generate connection map consisting of highest measured A connecting to point
    
    ny, nx = potential.shape[-2], potential.shape[-1]

    original_shape = potential.shape
    if len(original_shape) == 2:
        potential = potential.unsqueeze(0)
    potential = potential.reshape(-1, ny, nx)

    observed_potential = potential[..., ny // 2, :]
    peak_position = torch.abs(observed_potential).argmax(dim=-1)
    peak_values = observed_potential.gather(-1, peak_position.unsqueeze(-1)).squeeze(-1)
    
    relative_potential = potential / peak_values[:, None, None]
    relative_potential = torch.where(torch.isnan(relative_potential), torch.inf, relative_potential)

    visited = torch.zeros_like(relative_potential)

    # initial strip first
    visited[..., ny // 2, :].scatter_(-1, peak_position.unsqueeze(-1), 1)

    while True:
        # find largest graph value amongst the nearest pixels
        peak_neighbor_values = F.max_pool2d(visited, 3, 1, padding=1)   
        
        # cap to the actual value at that spot
        visited_new = torch.minimum(peak_neighbor_values, relative_potential)

        # if neighbor is maximum visited, and this would be the new maximum, increase above current maximum
        peak_values = visited.flatten(-2).amax(dim=-1)[:, None, None]
        neighbor_is_maximum = (peak_values == peak_neighbor_values)
        greatest_increase_position = torch.where(neighbor_is_maximum, relative_potential, 0).flatten(-2).argmax(dim=-1, keepdim=True)
        # for greatest increase position neighboring current maximum, set to relative_potential at that position even if it is higher than the peak neighbor value
        # where there is no such position, this should have no effect
        visited_new.flatten(-2).scatter_(-1, greatest_increase_position, relative_potential.flatten(-2).gather(-1, greatest_increase_position))
    
        # only change if more than current value
        visited_new = torch.maximum(visited, visited_new)
        
        if torch.all(visited == visited_new):
            break
      
        visited = visited_new

    # occurs when there are 2+ cores embedded in an outer closed loop (e.g. merging flux rope)
    # pick biggest value that is less than peak
    interior_cutoff = torch.where((relative_potential > visited), visited, 0).flatten(-2).amax(dim=-1)

    # outermost open field line
    exterior_cutoff = (torch.concat([visited[..., 0, :],
                                 visited[..., -1, :],
                                 visited[..., :, 0],
                                 visited[..., :, -1]], dim=-1)).amax(dim=-1)
    
    cutoff = torch.maximum(interior_cutoff, exterior_cutoff)
    # cutoff = exterior_cutoff

    core_mask = visited > cutoff[..., None, None]

    return core_mask.reshape(original_shape)


#@torch.jit.script
def _reconstruct_map(magnetic_field_observed,
                               gas_pressure_observed,
                               pixel_width,
                               alpha,
                               resolution: int = 21,
                               poly_order: int = 3,
                               aspect_ratio: float = 1):
    original_resolution = magnetic_field_observed.shape[-2]
    
    # smooth
    filter_length = max(1, original_resolution // 10)
    if filter_length % 2 == 0:
        filter_length += 1
    magnetic_field_observed = F.avg_pool1d(magnetic_field_observed.reshape(-1, original_resolution, 3).transpose(-1, -2),
                                           kernel_size=filter_length,
                                           stride=1,
                                           padding=filter_length // 2,
                                           count_include_pad=False).transpose(-1, -2).reshape(magnetic_field_observed.shape[:-2] + (original_resolution, 3))
    gas_pressure_observed = F.avg_pool1d(gas_pressure_observed.reshape(-1, 1, original_resolution),
                                           kernel_size=filter_length,
                                           stride=1,
                                           padding=filter_length // 2,
                                           count_include_pad=False).reshape(gas_pressure_observed.shape[:-1] + (original_resolution,))

    # rescale
    magnetic_field_observed = F.interpolate(magnetic_field_observed.reshape(-1, original_resolution, 3).transpose(-1, -2),
                                            resolution, mode="linear", align_corners=True
                                            ).transpose(-1, -2).reshape(magnetic_field_observed.shape[:-2] + (resolution, 3))
    gas_pressure_observed = F.interpolate(gas_pressure_observed.reshape(-1, 1, original_resolution),
                                        resolution, mode="linear", align_corners=True
                                        ).reshape(gas_pressure_observed.shape[:-1] + (resolution,))
    
    dx = 1

    magnetic_potential_observed = torch.zeros_like(magnetic_field_observed[..., 1])
    magnetic_potential_observed[..., 1:] = torch.cumulative_trapezoid(-magnetic_field_observed[..., 1], dx=dx)
    scale_factor = torch.abs(magnetic_potential_observed.mean(dim=-1))
    magnetic_potential_observed = magnetic_potential_observed / scale_factor.unsqueeze(-1)
    magnetic_field_observed = magnetic_field_observed / scale_factor.unsqueeze(-1).unsqueeze(-1)
    gas_pressure_observed = gas_pressure_observed / scale_factor.unsqueeze(-1)

    # 1 mu_0 nPa = 1256.637 nT^2
    field_line_quantity_observed = -(((scipy.constants.mu_0 * 1256.637) / (1 - alpha)) * gas_pressure_observed\
                                    + torch.linalg.norm(magnetic_field_observed, dim=-1) ** 2 / 2 * (alpha / (1 - alpha))\
                                    + magnetic_field_observed[..., 2] ** 2 / 2)
    
    potential_peak = magnetic_potential_observed.gather(-1, torch.abs(magnetic_potential_observed).argmax(dim=-1, keepdim=True)).squeeze(-1)
    peak_sign = torch.sign(potential_peak)

    field_line_quantity_coeffs = _polyfit(magnetic_potential_observed, field_line_quantity_observed, poly_order)

    field_line_quantity_fit = _evaluate_fit_values(field_line_quantity_coeffs, magnetic_potential_observed, peak_sign)
    field_line_quantity_fit_min, field_line_quantity_fit_max = torch.aminmax(field_line_quantity_fit, dim=-1)
    error_fit = torch.sqrt(torch.mean((field_line_quantity_observed - field_line_quantity_fit) ** 2, dim=-1))
    error_fit = error_fit / (field_line_quantity_fit_max - field_line_quantity_fit_min)

    # require derivative to be positive at tail
    error_fit = torch.where(torch.sign(-field_line_quantity_coeffs[..., 1]) == peak_sign.squeeze(-1), error_fit, torch.inf)

    # error_fit = error_fit / torch.abs(field_line_quantity_observed.mean(dim=-1))

    d2_dx2 = _second_derivative
    second_derivative_x = d2_dx2(magnetic_potential_observed)

    magnetic_potential_observed = magnetic_potential_observed
    first_derivative_y_observed = magnetic_field_observed[..., 0]    # Bx
    laplacian_observed = _laplacian(magnetic_potential_observed,
                                            potential_peak,
                                            poly_order,
                                            field_line_quantity_coeffs)

    second_derivative_y_observed = laplacian_observed - second_derivative_x

    extra_steps = 10

    extrapolation_length = int(magnetic_field_observed.shape[-2] * aspect_ratio // 2)

    output_map = torch.nan * torch.empty(magnetic_field_observed.shape[:-2] + (extrapolation_length * 2 + 1, magnetic_potential_observed.shape[-1]),
                                            dtype=magnetic_field_observed.dtype, device=magnetic_field_observed.device)
    output_map[..., extrapolation_length, :] = magnetic_potential_observed
    
    for direction in (-1, 1):
        dy = dx / extra_steps * direction

        potential = magnetic_potential_observed
            
        first_derivative_y = first_derivative_y_observed
        
        second_derivative_y = second_derivative_y_observed

        for i in torch.arange(1, extrapolation_length * extra_steps + 1, device=potential.device):            
            height = i / (extrapolation_length * extra_steps)

            potential = _three_point_smooth(potential + first_derivative_y * dy + (1 / 2) * second_derivative_y * (dy ** 2), height)

            first_derivative_y = _three_point_smooth(first_derivative_y + second_derivative_y * dy, height)

            second_derivative_x = d2_dx2(potential)
            second_derivative_y = _laplacian(potential,
                                             potential_peak,
                                             poly_order,
                                             field_line_quantity_coeffs)\
                                        - second_derivative_x

            output_map[..., extrapolation_length + torch.ceil(i / extra_steps).to(torch.long) * direction, :] = potential

    binomial_kernel = torch.as_tensor([[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]],
                                      dtype=output_map.dtype, device=output_map.device) / 16
    output_map[..., 1:-1, 1:-1] = F.conv2d(output_map.reshape((int(np.prod(output_map.shape[:-2])), 1, *output_map.shape[-2:])),
                                              binomial_kernel.reshape(1, 1, 3, 3),
                                              padding=1)[..., 0, 1:-1, 1:-1]

    magnetic_potential = output_map * pixel_width[..., None, None] * (original_resolution / resolution) * scale_factor[..., None, None]
    magnetic_field_x = torch.gradient(output_map, dim=-2, edge_order=1)[0] * scale_factor[..., None, None]
    magnetic_field_y = -torch.gradient(output_map, dim=-1, edge_order=1)[0] * scale_factor[..., None, None]
    magnetic_field_z_fit = _polyfit(magnetic_potential_observed, magnetic_field_observed[..., 2], poly_order)
    magnetic_field_z = _evaluate_fit_values(magnetic_field_z_fit, output_map.flatten(-2), peak_sign).reshape(output_map.shape) * scale_factor[..., None, None]
    gas_pressure_fit = _polyfit(magnetic_potential_observed, gas_pressure_observed, poly_order)
    gas_pressure = _evaluate_fit_values(gas_pressure_fit, output_map.flatten(-2), peak_sign).reshape(output_map.shape) * scale_factor[..., None, None]

    # from assumption of 2D magnetic field (\partial/\partial z = 0):
    # \mu_0 j_x = {\partial B_z}/{\partial y}
    # \mu_0 j_y = {-\partial B_z}{\partial x}
    # \mu_0 j_z = -\del^2 A_z
    current_density_unit = (1e-9) / (scipy.constants.mu_0 * pixel_width[..., None, None])  # mA/m^2 if B is in nT and pixel width in km
    current_density_x = torch.gradient(magnetic_field_z, dim=-2, edge_order=1)[0] * current_density_unit
    current_density_y = torch.gradient(magnetic_field_z, dim=-1, edge_order=1)[0] * current_density_unit
    current_density_z = -(_laplacian(output_map.flatten(-2),
                                    potential_peak,
                                    poly_order,
                                    field_line_quantity_coeffs).reshape(output_map.shape) * scale_factor[..., None, None]) * current_density_unit

    core_mask = _generate_connection_map(output_map)
    
    return ReconstructedMap(error_fit,
                                     magnetic_potential,
                                     magnetic_field_x,
                                     magnetic_field_y,
                                     magnetic_field_z,
                                     current_density_x,
                                     current_density_y,
                                     current_density_z,
                                     gas_pressure,
                                     core_mask)


