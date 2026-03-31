# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tdse import *

# Grid
Nx =            EXPORT(20000 * 2, "Nx")
delta_x =       EXPORT(0.2 / 2, "delta_x")
shift_x =       -(Nx * delta_x) / 2
Lx =            Nx * delta_x

# Time evolution
delta_t =       EXPORT(0.05 / 2, "delta_t")
imag_delta_t =  EXPORT(0.1, "imag_delta_t")
itp_steps =     EXPORT(1000, "itp_steps")

# t-surf
Xi =            EXPORT((Lx / 2) * 0.9, "Xi")

# laser
E0 =            EXPORT(0.05, "E0")
omega0 =        EXPORT(0.057, "omega0")
nc =            EXPORT(6, "nc")
laser = cos2_laser_pulse(delta_t=delta_t, E0=E0, omega0=omega0, nc=nc)

# potential 
a0 =            EXPORT(2.0, "a0")
b0 =            EXPORT(1.0, "b0")
short_range_flag =    EXPORT(True, "short_range_flag")
short_range_sigma =   EXPORT(5.0, "short_range_sigma")
Vx =            lambda x: -b0 / np.sqrt(x * x + a0) * (np.exp(- x * x / (short_range_sigma ** 2)) if short_range_flag == True else 1.0)
Vx_absorb =     lambda x: -100j * pow((np.abs(x) - Xi) / (Lx / 2 - Xi), 8) * (np.abs(x) > Xi)


# Runtime Environment
xgrid = create_grid_data(Nx, delta_x, shift_x)
rt = py_create_buffer_1d(Nx, delta_x, delta_t, imag_delta_t, shift_x, \
    Vx, Vx_absorb, accuracy=2, boundary_condition=EXPORT("reflect", "boundary_condition"))
wave = py_itp_1d(rt, itp_steps)

# fig, ax = plt.subplots()
# ax.plot(xgrid, rt["V"].diagonal())
# ax.set(xlim=(-10, 10))
# plt.show()

psi_pos, psi_neg, k, psi_k = separate_momentum_components(wave, xgrid)

en = py_get_energy_1d(rt, wave)
p1 = py_get_kinetic_momentum_1d(rt, psi_pos)
p2 = py_get_kinetic_momentum_1d(rt, psi_neg)
print(f"Energy = {en}")
# print(f"Momentum (+) = {p1}")
# print(f"Momentum (-) = {p2}")
# print(py_get_energy_1d(rt, psi_neg))
# print(f"Momentum wave = {py_get_kinetic_momentum_1d(rt, wave)}")
# plt.plot(np.imag(psi_pos))
# plt.plot(np.imag(psi_neg))
# plt.plot(np.imag(wave))
# plt.show()

psi_odd = (wave - np.flip(wave)) / 2
psi_even = (wave + np.flip(wave)) / 2
psi_odd /= np.sqrt(np.vdot(psi_odd, psi_odd))
psi_even /= np.sqrt(np.vdot(psi_even, psi_even))

sign_H = build_sign_operator(rt["H"], zero_tol=1e-10)
# wave_sign = apply_sign_operator(sign_H, wave)

sign_H["negative_count"]
plt.plot(sign_H["negative_evecs"][:, 2], label="sgn(H) negative evec 1")
plt.show()

# plt.plot(np.abs(wave), label="wave")
# plt.plot(np.abs(wave_sign), label="sgn(H) wave")
# plt.legend()
# plt.show()

# print(f"sgn(H) representation = {sign_H['kind']}")