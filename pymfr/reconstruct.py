import scipy.integrate
import scipy.constants
from scipy.signal import resample_poly
import numpy as np


class ReconstructedCrossSection():
    def __init__(self, magnetic_potential, magnetic_field_x, magnetic_field_y, magnetic_field_z, gas_pressure, current_density_x, current_density_y, current_density_z) -> None:
        self.magnetic_potential = magnetic_potential
        self.magnetic_field_x = magnetic_field_x
        self.magnetic_field_y = magnetic_field_y
        self.magnetic_field_z = magnetic_field_z
        self.gas_pressure = gas_pressure
        self.current_density_x = current_density_x
        self.current_density_y = current_density_y
        self.current_density_z = current_density_z


def reconstruct_cross_section(magnetic_field_observed, gas_pressure_observed, cross_section_sample_spacing, poly_order=3, aspect_ratio=1):
    """
    GS reconstruction of MFR cross section (Hu et al. 2001)
    Takes a variable number of samples. Recommended 15 or 2ss1 samples for numerical stability of the solver & a square grid.
    
    :param magnetic_field_observed: Magnetic field vectors in nT, shape (N, 3).
    :param gas_pressure_observed: Gas pressure in nPa, shape (N, 3)
    :param cross_section_sample_spacing: Spacing between samples in meters.
    :return: A ReconstructedCrossSection object.
    """

    dx = 1
    
    magnetic_potential_observed = scipy.integrate.cumulative_trapezoid(-magnetic_field_observed[:, 1], dx=dx, initial=0)
    first_derivative_y_observed = magnetic_field_observed[:, 0]    # Bx
    # mu_0 nPa/nT^2 = 1256.637062
    field_line_quantity_observed = (scipy.constants.mu_0 * gas_pressure_observed * 1256.637062 + magnetic_field_observed[:, 2] ** 2 / 2)
    
    potential_peak = magnetic_potential_observed[np.abs(magnetic_potential_observed).argmax()]
    peak_sign = np.sign(potential_peak)

    field_line_quantity_fit = np.poly1d(np.polyfit(magnetic_potential_observed, field_line_quantity_observed, poly_order))
    assert np.sign(field_line_quantity_fit.deriv()(0)) == peak_sign, "tail is not decreasing"

    def laplacian(potential):
        return -_evaluate_fit_values(field_line_quantity_fit.deriv(), potential, peak_sign)

    d2_dx2 = _second_derivative
    second_derivative_x = d2_dx2(magnetic_potential_observed)
    second_derivative_y = laplacian(magnetic_potential_observed) - second_derivative_x

    magnetic_potential_observed = magnetic_potential_observed
    first_derivative_y_observed = first_derivative_y_observed
    second_derivative_y_observed = second_derivative_y

    extra_steps = 10

    extrapolation_length = int(len(magnetic_field_observed) * aspect_ratio // 2)

    cross_section = np.nan * np.zeros((extrapolation_length * 2 + 1,
                                            len(magnetic_potential_observed)))
    cross_section[extrapolation_length] = magnetic_potential_observed
    
    for direction in (-1, 1):
        dy = dx / extra_steps * direction

        extension = np.nan * np.zeros((extrapolation_length * extra_steps + 1,
                                            len(magnetic_potential_observed)))
        extension[0] = magnetic_potential_observed

            
        first_derivative_y = np.nan * np.zeros((extrapolation_length * extra_steps + 1,
                                                len(first_derivative_y_observed)))
        first_derivative_y[0] = first_derivative_y_observed
        
        second_derivative_y = np.nan * np.zeros((extrapolation_length * extra_steps + 1,
                                                len(second_derivative_y_observed)))
        second_derivative_y[0] = second_derivative_y_observed

        for i in range(1, len(extension)):            
            height = i / len(extension)

            extension[i] = extension[i - 1]\
              + first_derivative_y[i - 1, :] * dy\
              + (1 / 2) * second_derivative_y[i - 1, :] * (dy ** 2)
            extension[i] = _three_point_smooth(extension[i], height)

            first_derivative_y[i] = first_derivative_y[i - 1] + second_derivative_y[i - 1] * dy
            first_derivative_y[i] = _three_point_smooth(first_derivative_y[i], height)

            second_derivative_x = d2_dx2(extension[i])
            second_derivative_y[i] = laplacian(extension[i]) - second_derivative_x

        extension = extension[1:]
        extension = extension.reshape(extension.shape[0] // extra_steps, extra_steps, extension.shape[1]).mean(axis=1)
        
        if direction == 1:
          cross_section[extrapolation_length + 1:] = extension
        else:
          cross_section[:extrapolation_length] = extension[::-1]

    magnetic_potential = cross_section * cross_section_sample_spacing
    magnetic_field_x = np.gradient(cross_section, axis=0)
    magnetic_field_y = -np.gradient(cross_section, axis=1)
    magnetic_field_z_fit = np.poly1d(np.polyfit(magnetic_potential_observed, magnetic_field_observed[:, 2], poly_order))
    magnetic_field_z = _evaluate_fit_values(magnetic_field_z_fit, cross_section, peak_sign)
    gas_pressure_fit = np.poly1d(np.polyfit(magnetic_potential_observed, gas_pressure_observed, poly_order))
    gas_pressure = _evaluate_fit_values(gas_pressure_fit, cross_section, peak_sign)

    # TODO: Fix the units of current density
    current_density_x = np.gradient(magnetic_field_z, cross_section_sample_spacing, axis=0) / scipy.constants.mu_0
    current_density_y = -np.gradient(magnetic_field_z, cross_section_sample_spacing, axis=1) / scipy.constants.mu_0
    current_density_z = _evaluate_fit_values(field_line_quantity_fit.deriv(), cross_section, peak_sign) / scipy.constants.mu_0

    return ReconstructedCrossSection(magnetic_potential, magnetic_field_x, magnetic_field_y, magnetic_field_z,
                                     gas_pressure, current_density_x, current_density_y, current_density_z)


def _evaluate_fit_values(polynomial_fit, potential, peak_sign):
    tail_coefficient_left = polynomial_fit(0)
    tail_exponent_left = polynomial_fit(0) / tail_coefficient_left 

    return np.where(potential * peak_sign >= 0,
                                    polynomial_fit(potential),
                                    tail_coefficient_left * tail_exponent_left * np.exp(tail_exponent_left * potential))


def _three_point_smooth(solution, height):
    smoothed = np.copy(solution)
    center_weight = 1 - height / 3
    edge_weight = (1 - center_weight) / 2
    smoothed[1:-1] = center_weight * solution[1:-1] + edge_weight * (solution[:-2] + solution[2:])
    return smoothed



def _second_derivative(solution, dx=1):
    result = np.zeros_like(solution)
    result[1:-1] = solution[2:] - 2 * solution[1:-1] + solution[:-2]
    result[0] = 2 * solution[0] - 5 * solution[1] + 4 * solution[2] - solution[3]
    result[-1] = 2 * solution[-1] - 5 * solution[-1 - 1] + 4 * solution[-1 - 2] - solution[-1 - 3]
    return result / dx ** 2