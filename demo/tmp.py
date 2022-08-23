import pathlib

import cdflib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import scipy.constants
import torch
from cdasws import CdasWs

time = ["2016-01-31T14:55:00Z", "2016-01-31T20:50:00Z"]

file_name = f"./data/wind_{time[0]}_{time[1]}_1min_nonlin.npz"
if not pathlib.Path(file_name).exists():
    cdas = CdasWs()

    status, data = cdas.get_data('WI_H1_SWE', ['Proton_VX_nonlin',
                                               'Proton_VY_nonlin',
                                               'Proton_VZ_nonlin',
                                               'Proton_Np_nonlin',
                                               'Proton_W_nonlin'],
                                 time[0], time[1])

    datetime = cdflib.epochs.CDFepoch().to_datetime(data.Epoch.values, to_np=True)

    velocity = np.column_stack([data.Proton_VX_nonlin.values,
                                data.Proton_VY_nonlin.values,
                                data.Proton_VZ_nonlin.values])
    velocity[np.any(velocity == data.Proton_VX_nonlin.FILLVAL, axis=1)] = np.nan
    velocity[np.any(velocity == data.Proton_VY_nonlin.FILLVAL, axis=1)] = np.nan
    velocity[np.any(velocity == data.Proton_VZ_nonlin.FILLVAL, axis=1)] = np.nan
    velocity = pd.DataFrame(velocity, index=pd.DatetimeIndex(datetime)).interpolate()
    velocity = velocity.resample("60s").mean().shift(0.5, freq="60s").interpolate()
    times = velocity.index.values

    density = data.Proton_Np_nonlin.values
    density[density == data.Proton_Np_nonlin.FILLVAL] = np.nan
    density = pd.Series(density, index=pd.DatetimeIndex(datetime)).interpolate()
    density = density.resample("60s").mean().shift(0.5, freq="60s").interpolate()

    temperature = data.Proton_W_nonlin.values
    temperature[temperature == data.Proton_W_nonlin.FILLVAL] = np.nan
    temperature = (temperature * 1e3) ** 2 * scipy.constants.m_p / (2 * scipy.constants.Boltzmann) / 1e6
    temperature = pd.Series(temperature, index=pd.DatetimeIndex(datetime)).interpolate()
    temperature = temperature.resample("60s").mean().shift(0.5, freq="60s").interpolate()

    status, data = cdas.get_data('WI_H0_MFI', ['BGSE'], time[0], time[1])

    magnetic_field = data.BGSE.values
    magnetic_field[np.any(magnetic_field == data.BGSE.FILLVAL, axis=1), :] = np.nan
    datetime = cdflib.epochs.CDFepoch().to_datetime(data.Epoch.values, to_np=True)
    magnetic_field = pd.DataFrame(magnetic_field, index=pd.DatetimeIndex(datetime)).interpolate()

    magnetic_field = magnetic_field.truncate(times[0], times[-1])
    density = density.truncate(magnetic_field.index[0], magnetic_field.index[-1]).values
    temperature = temperature.truncate(magnetic_field.index[0], magnetic_field.index[-1]).values
    velocity = velocity.truncate(magnetic_field.index[0], magnetic_field.index[-1]).values
    times = magnetic_field.index.values

    assert len(magnetic_field) == len(velocity) == len(density)

    np.savez_compressed(file_name, magnetic_field=magnetic_field, velocity=velocity, density=density, temperature=temperature, times=times)

data = np.load(file_name)

magnetic_field = torch.as_tensor(data["magnetic_field"], dtype=torch.float32)
velocity = torch.as_tensor(data["velocity"], dtype=torch.float32)
density = torch.as_tensor(data["density"], dtype=torch.float32)
temperature = torch.as_tensor(data["temperature"], dtype=torch.float32)
times = data["times"]

from pymfr.frame import estimate_ht_frame
from pymfr.axis import minimize_rdiff

electric_field = -torch.cross(velocity, magnetic_field)
frame = estimate_ht_frame(magnetic_field, electric_field)
# frame = velocity.mean(dim=0)

gas_pressure = scipy.constants.Boltzmann * (density * 1e6) * (temperature * 1e6) * 1e9
# gas_pressure = torch.zeros_like(gas_pressure)

trial_altitudes = np.linspace(0, 90, 180)
trial_azimuths = np.linspace(0, 360, 720)
trial_axes = [torch.tensor([np.sin(np.deg2rad(altitude)) * np.cos(np.deg2rad(azimuth)),
                            np.sin(np.deg2rad(altitude)) * np.sin(np.deg2rad(azimuth)),
                            np.cos(np.deg2rad(altitude))], dtype=torch.float32)
              for altitude in trial_altitudes
              for azimuth in trial_azimuths]
trial_axes.append(torch.tensor([0, 0, 1], dtype=torch.float32))
trial_axes = torch.row_stack(trial_axes)

from pymfr.axis import calculate_residue_map
residue_map = calculate_residue_map(magnetic_field, gas_pressure, frame, trial_axes)[:-1].reshape((180, 720)).numpy()
residue_map[residue_map == np.inf] = 1

best_axis, best_rdiff = minimize_rdiff(magnetic_field.cuda(),
                                       gas_pressure.cuda(),
                                       frame.cuda(),
                                       trial_axes.cuda())
best_axis = best_axis.cpu()
best_rdiff = best_rdiff.cpu()
best_axis, best_rdiff
