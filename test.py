# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tdse import *

# Grid
Nx = 3600
delta_x = 0.2
shift_x = -(Nx * delta_x) / 2
Lx = Nx * delta_x

# Time evolution
delta_t = 0.05
imag_delta_t = 0.1

# t-surf
Xi = 200

# laser
laser1 = cos2_laser_pulse(delta_t=delta_t, E0=0.053, omega0=0.057, nc=6)
dc = dc_bias(delta_t, Edc=0.0, last_time=2 * (laser1.t_max - laser1.t_min))
laser = combine_light_field(laser1, dc)

# laser.display()

# E_dc
# laser = dc_bias(Edc=0.01, last_time=200.0)

# potential 
Vx = lambda x: -1.0 / np.sqrt(x * x + 1.0)
Vx_absorb = lambda x: -100j * pow((np.abs(x) - Xi) / (Lx / 2 - Xi), 8) * (np.abs(x) > Xi)

# tdse
world, xs = create_physical_world_1d(Nx, delta_x, shift_x, Vx, Vx_absorb)
buffer = create_runtime_buffer_1d(world, delta_t, imag_delta_t)
wave = get_ground_state_1d(world, buffer, 500)
tsurf_res = tdse_fd1d_tsurf(world, buffer, wave, light_field=laser, Xi=Xi)
ks, Pk = tsurf_1d(tsurf_res, light_field=laser, k_min=-3.0, k_max=3.0, Xi=Xi)

# plot the results
plt.plot(np.sign(ks) * pow(ks, 2) / 2, Pk, lw=0.5)
plt.yscale('log')
plt.ylim(1e-15, 1e-4)
plt.yticks([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
plt.show()