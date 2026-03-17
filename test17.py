# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tdse import *

# Grid
Nx =            EXPORT(3600 * 2, "Nx")
delta_x =       EXPORT(0.2 / 2, "delta_x")
shift_x =       -(Nx * delta_x) / 2
Lx =            Nx * delta_x

# Time evolution
delta_t =       EXPORT(0.05 / 2, "delta_t")
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

# get eigenstates
ek_list = [-0.669, -0.27, -0.15, -0.09, -0.063, -0.045, -0.034, -0.026, -0.0217, -0.017]    # for a0 = 1.0
eigen_waves = []
for i in range(0, 6):
    ek = ek_list[i]
    imag_delta_t_tmp = get_imag_delta_t_from_ek(ek)
    buffer_tmp = create_runtime_buffer_1d(world, delta_t, imag_delta_t_tmp)
    e_wave = get_ground_state_1d(buffer_tmp, itp_steps)
    eigen_waves.append(e_wave)

# tdse-hg
accel, pos, accel_free, pos_free, accel_bound, pos_bound = tdse_fd1d_hg_analytical(world, buffer, wave, light_field=laser_all, Xi=Xi, bound_states=eigen_waves)

plt.figure()
plt.plot(laser.get_ts(), accel, label=" expect total")
plt.plot(laser.get_ts(), accel_free, label=" expect free")
plt.plot(laser.get_ts(), accel_bound, label=" expect bound")
plt.legend()
plt.show()

# harmonic spectrum
max_display_n = EXPORT(30, "max_display_n")
display_spacing_n = EXPORT(2, "display_spacing_n")
display_start_n = EXPORT(1, "display_start_n")

pos_expect_spectrum, ks = get_pos_expect_spectrum_1d(laser.get_ts(), pos, max_k= max_display_n * omega0)
pos_expect_spectrum_free, _ = get_pos_expect_spectrum_1d(laser.get_ts(), pos_free, max_k= max_display_n * omega0)
pos_expect_spectrum_bound, _ = get_pos_expect_spectrum_1d(laser.get_ts(), pos_bound, max_k= max_display_n * omega0)
pos_expect_spectrum_residual, _ = get_pos_expect_spectrum_1d(laser.get_ts(), pos - pos_free - pos_bound, max_k= max_display_n * omega0)


plt.figure()
plt.plot(ks / omega0, np.pow(np.abs(pos_expect_spectrum), 2), label="pos expect total")
plt.plot(ks / omega0, np.pow(np.abs(pos_expect_spectrum_free), 2), label="pos expect free")
plt.plot(ks / omega0, np.pow(np.abs(pos_expect_spectrum_bound), 2), label="pos expect bound")
plt.plot(ks / omega0, np.pow(np.abs(pos_expect_spectrum_residual), 2), label="pos expect residual")

plt.xticks(range(display_start_n, max_display_n, display_spacing_n))
plt.grid(True, alpha=0.2)
plt.yscale('log')
plt.ylim(1e-15, 1e3)
plt.legend()
plt.show()