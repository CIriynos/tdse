# -*- coding: utf-8 -*-
import os
import numpy as np
from ctypes import *
from ctypes import CDLL
import matplotlib.pyplot as plt
import sys

if os.name == "nt":
    lib = CDLL("./interface.dll", winmode=0)
else:
    lib = cdll.LoadLibrary("./interface.so")

lib.create_physical_world_1d.restype = c_void_p
lib.create_runtime_buffer_1d.restype = c_void_p
lib.get_ground_state_1d.restype = c_void_p
lib.get_energy_1d.restype = c_double
lib.tdse_laser_fd1d_onestep.restype = c_double
lib.get_norm_1d.restype = POINTER(c_double)
lib.get_wave_value_1d.restype = POINTER(c_double)
lib.get_wave_1diff_value_1d.restype = POINTER(c_double)

def create_physical_world_1d(
    Nx, delta_x, shift_x,   # x-grid
    potential,
    absorption_potential
    ):

    xs = [shift_x + delta_x * i for i in range(0, Nx)]
    Vx = [potential(x) for x in xs]
    Vx_absorb_real = [np.real(absorption_potential(x)) for x in xs]
    Vx_absorb_imag = [np.imag(absorption_potential(x)) for x in xs]

    c_array_t = c_double * Nx
    Vx__ = c_array_t()
    Vx_absorb_real__ = c_array_t()
    Vx_absorb_imag__ = c_array_t()
    for i in range(0, Nx):
        Vx__[i] = c_double(Vx[i])
        Vx_absorb_real__[i] = c_double(Vx_absorb_real[i])
        Vx_absorb_imag__[i] = c_double(Vx_absorb_imag[i])

    world = lib.create_physical_world_1d(c_int(Nx), c_double(delta_x), c_double(shift_x),
        Vx__, Vx_absorb_real__, Vx_absorb_imag__)
    return world, xs

def test(world):
    return lib.test(c_void_p(world))

def create_runtime_buffer_1d(world, delta_t, imag_delta_t):
    buffer = lib.create_runtime_buffer_1d(c_void_p(world), c_double(delta_t), c_double(imag_delta_t))
    return buffer

def get_ground_state_1d(world, buffer, time_steps):
    wavefunc = lib.get_ground_state_1d(c_void_p(world), c_void_p(buffer), c_int(time_steps))
    return wavefunc

def get_energy_1d(buffer, wavefunc):
    return lib.get_energy_1d(c_void_p(buffer), c_void_p(wavefunc))

def get_norm_1d(buffer, wavefunc):
    res = lib.get_norm_1d(c_void_p(buffer), c_void_p(wavefunc))
    return res[0] + 1j * res[1]

def get_wave_value_1d(world, wavefunc, x_pos):
    res = lib.get_wave_value_1d(c_void_p(world), c_void_p(wavefunc), c_double(x_pos))
    return res[0] + 1j * res[1]

def tdse_laser_fd1d_onestep(buffer, wavefunc, At):
    lib.tdse_laser_fd1d_onestep(c_void_p(buffer), c_void_p(wavefunc), c_double(At))

def get_wave_1diff_value_1d(world, wavefunc, x_pos):
    res = lib.get_wave_1diff_value_1d(c_void_p(world), c_void_p(wavefunc), c_double(x_pos))
    return res[0] + 1j * res[1]


###################################

class light_field:
    def __init__(self, delta_t, t_min, t_max):
        self.t_min = t_min
        self.t_max = t_max
        self.delta_t = delta_t
        self.Et = lambda t: 0.0  # empty Et
      
    def get_ts(self):
        return np.array([self.t_min + i * self.delta_t for i in range(0, round(((self.t_max - self.t_min)) / self.delta_t))])

    def get_steps(self):
        ts = self.get_ts(self.delta_t)
        return len(ts)

    def get_Et_data(self):
        vectorized_Et = np.vectorize(self.Et)
        ts = self.get_ts()
        return vectorized_Et(ts)
    
    def get_At_data(self):
        Et_data = self.get_Et_data()
        return np.cumsum(Et_data) * (self.delta_t)

    def display(self):
        ts = self.get_ts()
        plt.plot(ts, self.get_Et_data(ts))


def combine_light_field(light1, light2):
    t_min = min(light1.t_min, light2.t_min)
    t_max = max(light1.t_max, light2.t_max)
    delta_t = light1.delta_t
    assert(light1.delta_t == light2.delta_t)
    light = light_field(delta_t, t_min, t_max)
    light.Et = lambda t: light1.Et(t) + light2.Et(t)
    return light


class dc_bias(light_field):
    def __init__(self, delta_t, Edc, last_time, t_shift=0.0):
        super().__init__(
            delta_t = delta_t,
            t_min = t_shift,
            t_max = t_shift + last_time
        )
        self.Et = lambda t: Edc * (t > self.t_min and t < self.t_max)


class cos2_laser_pulse(light_field):
    def __init__(self, delta_t, E0, omega0, nc, phi0=0.0, t_shift=0.0):
        super().__init__(
            delta_t = delta_t,
            t_min = t_shift,
            t_max = t_shift + (2 * nc * np.pi / omega0)
        )
        self.E0 = E0
        self.omega0 = omega0
        self.nc = nc
        self.phi0 = phi0
        self.Tp = 2 * nc * np.pi / omega0
        self.Et = lambda t: E0 * np.sin(omega0 * t / nc / 2) * np.cos(omega0 * t + phi0) * (t > self.t_min and t < self.t_max)


class sin2_laser_pulse_At(light_field):
    def __init__(self, delta_t, A0, omega0, nc, phi0=0.0, t_shift=0.0):
        super().__init__(
            delta_t = delta_t,
            t_min = t_shift,
            t_max = t_shift + (2 * nc * np.pi / omega0)
        )
        self.omega0 = omega0
        self.nc = nc
        self.phi0 = phi0
        self.Tp = 2 * nc * np.pi / omega0
        self.At = lambda t: A0 * np.sin(omega0 * t / nc / 2) * np.sin(omega0 * t + phi0) * (t > self.t_min and t < self.t_max)

    def get_At_data(self):
        ts = self.get_ts()
        vectorized_At = np.vectorize(self.At)
        return vectorized_At(ts)




###########################################3

def tdse_fd1d_tsurf(world, buffer, wave, light_field, Xi):
    ts = light_field.get_ts()
    At_data = light_field.get_At_data()
    X_pos_vals = np.zeros(len(ts), dtype=complex)
    X_neg_vals = np.zeros(len(ts), dtype=complex)
    X_pos_dvals = np.zeros(len(ts), dtype=complex)
    X_neg_dvals = np.zeros(len(ts), dtype=complex)

    for (i, t) in enumerate(ts):
        tdse_laser_fd1d_onestep(buffer, wave, At=At_data[i])

        X_pos_vals[i] = (get_wave_value_1d(world, wave, Xi))
        X_neg_vals[i] = (get_wave_value_1d(world, wave, -Xi))
        X_pos_dvals[i] = (get_wave_1diff_value_1d(world, wave, Xi))
        X_neg_dvals[i] = (get_wave_1diff_value_1d(world, wave, -Xi))
        
        # logging
        if i % 500 == 0:
            energy = get_energy_1d(buffer, wave)
            norm = get_norm_1d(buffer, wave)
            print(f"[TDSE-fd1d] energy = {energy}, norm = {norm}")

    return [X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals]


def tsurf_1d(tsurf_res, light_field, k_min, k_max, Xi, sampling_num=500):
    print(f"[TSURF-1d] Excecuting t-surf(1d) ...")
    X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals = tsurf_res
    ts = light_field.get_ts()
    At_data = light_field.get_At_data()
    ks = np.linspace(k_min, k_max, num=500)
    b1k = np.zeros(len(ks), dtype=complex)
    b2k = np.zeros(len(ks), dtype=complex)
    alpha = np.cumsum(At_data) * light_field.delta_t
    hanning_window = np.vectorize(lambda t: 0.5 * (1 - np.cos(2 * np.pi * t / ts[-1])))

    foo1 = np.zeros(len(ts), dtype=complex)
    foo2 = np.zeros(len(ts), dtype=complex)
    
    # for (i, k) in enumerate(ks):
    #     for (j, t) in enumerate(ts):
    #         b1k[i] += hanning_window(t, ts[-1]) * (light_field.delta_t / np.sqrt(2 * np.pi)) * np.exp(1.0j * t * k * k / 2) * np.exp(-1.0j * k * (Xi - alpha[j])) * ((0.5 * k + At_data[j]) * X_pos_vals[j] - 0.5j * X_pos_dvals[j])
    #         b2k[i] += hanning_window(t, ts[-1]) * (-light_field.delta_t / np.sqrt(2 * np.pi)) * np.exp(1.0j * t * k * k / 2) * np.exp(-1.0j * k * (-Xi - alpha[j])) * ((0.5 * k + At_data[j]) * X_neg_vals[j] - 0.5j * X_neg_dvals[j])

    for (i, k) in enumerate(ks):
        foo1 = hanning_window(ts) * (light_field.delta_t / np.sqrt(2 * np.pi)) * np.exp(1.0j * ts * k * k / 2) * np.exp(-1.0j * k * (Xi - alpha)) * ((0.5 * k + At_data) * X_pos_vals - 0.5j * X_pos_dvals)
        foo2 = hanning_window(ts) * (-light_field.delta_t / np.sqrt(2 * np.pi)) * np.exp(1.0j * ts * k * k / 2) * np.exp(-1.0j * k * (-Xi - alpha)) * ((0.5 * k + At_data) * X_neg_vals - 0.5j * X_neg_dvals)
        b1k[i] = np.sum(foo1)
        b2k[i] = np.sum(foo2)

    Pk = np.abs(b1k + b2k) ** 2
    return ks, Pk