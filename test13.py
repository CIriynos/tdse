from tdse import *
from multiprocessing import Pool
import re

# Test 13
# 

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tdse import *

# Grid
Nx =            EXPORT(3600 * 1, "Nx")
delta_x =       EXPORT(0.2 / 1, "delta_x")
shift_x =       -(Nx * delta_x) / 2
Lx =            Nx * delta_x

# Time evolution
delta_t =       EXPORT(0.05 / 1, "delta_t")
imag_delta_t =  EXPORT(0.1, "imag_delta_t")
itp_steps =     EXPORT(1000, "itp_steps")

# t-surf & absorbing boundary
Xi =            EXPORT(Nx * delta_x / 2 * 0.8, "Xi")

# laser
E0 =            EXPORT(0.03, "E0")
omega0 =        EXPORT(0.05, "omega0")
nc =            EXPORT(8, "nc")
Edc =           EXPORT(0.0005 * 0, "Edc")
E0_sh =         EXPORT(0.0001 * 0, "E0_sh")
sh_phi =        EXPORT(0.0, "sh_phi")
freq_change_rate = EXPORT(2, "freq_change_rate")
laser = cos2_laser_pulse(delta_t=delta_t, E0=E0, omega0=omega0, nc=nc)
dc_bias = dc_bias_smooth(delta_t, Edc, laser.get_duration())
laser_all = combine_light_field(laser, dc_bias)
sh_field = cos2_laser_pulse(delta_t=delta_t, E0=E0_sh, omega0=omega0 * freq_change_rate, nc=nc * freq_change_rate, phi0=sh_phi)
laser_all = combine_light_field(laser_all, sh_field)

# potential 
a0 =            EXPORT(1.0, "a0")
Vx =            lambda x: -1.0 / np.sqrt(x * x + a0)
Vx_absorb =     lambda x: -100j * pow((np.abs(x) - Xi) / (Lx / 2 - Xi), 8) * (np.abs(x) > Xi) * 0.1


# enviroment
world, xs = create_physical_world_1d(Nx, delta_x, shift_x, Vx, Vx_absorb)
buffer = create_runtime_buffer_1d(world, delta_t, imag_delta_t)
wave = get_ground_state_1d(buffer, itp_steps)
init_energy = get_energy_1d(buffer, wave)
print(f"initial energy = {init_energy}")


# tdse-hg-tsurf
accel, pos, tsurf_res = tdse_fd1d_hg_tsurf(world, buffer, wave, light_field=laser_all, Xi=Xi)


# harmonic spectrum
n_cut_off_estim = math.floor((-init_energy + 3.17 * (E0 ** 2.0 / (4.0 * (omega0 ** 2.0)))) / omega0)
max_display_n = EXPORT(30, "max_display_n")
display_spacing_n = EXPORT(2, "display_spacing_n")
display_start_n = EXPORT(1, "display_start_n")

hg1, hg2, ks = get_hg_spectrum_1d(laser.get_ts(), accel, pos, max_k= max_display_n * omega0)
pos_expect_spectrum = get_pos_expect_spectrum_1d(laser.get_ts(), pos, max_k= max_display_n * omega0)

plt.figure()
plt.plot(ks / omega0, hg1, label="accel")
plt.plot(ks / omega0, hg2, label="pos")
# plt.plot(ks / omega0, np.pow(np.abs(pos_expect_spectrum), 2), label="pos expect")
plt.xticks(range(display_start_n, max_display_n, display_spacing_n))
plt.grid(True, alpha=0.2)
plt.yscale('log')
plt.ylim(1e-15, 1e3)
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