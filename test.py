# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tdse import *

# Grid
Nx =            EXPORT(3600, "Nx")
delta_x =       EXPORT(0.2, "delta_x")
shift_x =       -(Nx * delta_x) / 2
Lx =            Nx * delta_x

# Time evolution
delta_t =       EXPORT(0.05, "delta_t")
imag_delta_t =  EXPORT(0.1, "imag_delta_t")
itp_steps =     EXPORT(1000, "itp_steps")

# t-surf
Xi =            EXPORT(200, "Xi")

# laser
E0 =            EXPORT(0.05, "E0")
omega0 =        EXPORT(0.057, "omega0")
nc =            EXPORT(6, "nc")
Edc =           EXPORT(0.0, "Edc")
laser = cos2_laser_pulse(delta_t=delta_t, E0=E0, omega0=omega0, nc=nc)
dc_bias = dc_bias(delta_t, Edc, laser.get_duration())
laser_all = combine_light_field(laser, dc_bias)

# potential 
a0 =            EXPORT(1.0, "a0")
Vx =            lambda x: -1.0 / np.sqrt(x * x + a0)
Vx_absorb =     lambda x: -100j * pow((np.abs(x) - Xi) / (Lx / 2 - Xi), 8) * (np.abs(x) > Xi)


# enviroment
world, xs = create_physical_world_1d(Nx, delta_x, shift_x, Vx, Vx_absorb)
buffer = create_runtime_buffer_1d(world, delta_t, imag_delta_t)
wave = get_ground_state_1d(buffer, itp_steps)
init_energy = get_energy_1d(buffer, wave)

# tdse-hg-tsurf
accel, pos, tsurf_res = tdse_fd1d_hg_tsurf(world, buffer, wave, light_field=laser_all, Xi=Xi)

# harmonic spectrum
n_cut_off_estim = math.floor((-init_energy + 3.17 * (E0 ** 2.0 / (4.0 * (omega0 ** 2.0)))) / omega0)
print(f"n_cut_off = {n_cut_off_estim}")
hg1, hg2, ks = get_hg_spectrum_1d(laser.get_ts(), accel, pos, max_k=(30) * omega0)

plt.figure()
# plt.plot(ks / omega0, hg1, label="accel")
plt.plot(ks / laser.omega0, hg2, label="pos")
plt.xticks(range(1, (30), 2))
plt.grid(True, alpha=0.2)
plt.yscale('log')
plt.ylim(1e-15, 1e2)
plt.legend()
plt.show()

# # T-SURFF
# ks, Pk = tsurf_1d(tsurf_res, light_field=laser, k_min=-3.0, k_max=3.0, Xi=Xi, sampling_num=900)

# plt.figure()
# plt.plot(np.sign(ks) * pow(ks, 2) / 2, Pk, lw=0.5)
# plt.yscale('log')
# plt.xlim(-3.5, 3.5)
# plt.ylim(1e-15, 1e-4)
# plt.yticks([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
# plt.show()