# -*- coding: utf-8 -*-
import math
import os
import numpy as np
from ctypes import *
from ctypes import CDLL
import matplotlib.pyplot as plt
import sys
from scipy.signal.windows import chebwin

if os.name == "nt":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib = CDLL(current_dir + "/interface.dll", winmode=0)
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib = cdll.LoadLibrary(current_dir + "/interface.so")

lib.create_physical_world_1d.restype = c_void_p
lib.create_runtime_buffer_1d.restype = c_void_p
lib.get_ground_state_1d.restype = c_void_p
lib.tdse_laser_fd1d_onestep.restype = c_double
lib.get_energy_1d.restype = c_double
lib.get_norm_1d.restype = POINTER(c_double)
lib.get_pos_expect_1d.restype = c_double
lib.get_accel_expect_1d.restype = c_double
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

def get_ground_state_1d(buffer, time_steps):
    wavefunc = lib.get_ground_state_1d(c_void_p(buffer), c_int(time_steps))
    return wavefunc

def get_energy_1d(buffer, wavefunc):
    return lib.get_energy_1d(c_void_p(buffer), c_void_p(wavefunc))

def get_norm_1d(world, wavefunc):
    res = lib.get_norm_1d(c_void_p(world), c_void_p(wavefunc))
    return res[0] + 1j * res[1]

def get_pos_expect_1d(world, wavefunc):
    res = lib.get_pos_expect_1d(c_void_p(world), c_void_p(wavefunc))
    return res

def get_accel_expect_1d(world, wavefunc):
    res = lib.get_accel_expect_1d(c_void_p(world), c_void_p(wavefunc))
    return res

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
    
    def get_duration(self):
        return self.t_max - self.t_min

    def get_Et_data(self):
        vectorized_Et = np.vectorize(self.Et)
        ts = self.get_ts()
        return vectorized_Et(ts)
    
    def get_At_data(self):
        Et_data = self.get_Et_data()
        return -np.cumsum(Et_data) * (self.delta_t)

    def display(self):
        ts = self.get_ts()
        p = plt.plot(ts, self.get_Et_data())
        return p


def combine_light_field(light1, light2):
    t_min = min(light1.t_min, light2.t_min)
    t_max = max(light1.t_max, light2.t_max)
    delta_t = light1.delta_t
    assert(light1.delta_t == light2.delta_t)
    light = light_field(delta_t, t_min, t_max)
    light.Et = lambda t: light1.Et(t) + light2.Et(t)
    return light

def append_light_field(light1, light2):
    t_min = light1.t_min
    t_max = light1.t_max + (light2.t_max - light2.t_min)
    delta_t = light1.delta_t
    assert(light1.delta_t == light2.delta_t)
    light = light_field(delta_t, t_min, t_max)
    light.Et = lambda t: (light1.Et(t)) if (t >= light1.t_min and t <= light1.t_max) \
        else (light2.Et(light2.t_min + (t - light1.t_max))) * (t >= t_min and t <= t_max)
    return light


class dc_bias(light_field):
    def __init__(self, delta_t, Edc, last_time, t_shift=0.0):
        super().__init__(
            delta_t = delta_t,
            t_min = t_shift,
            t_max = t_shift + last_time
        )
        self.Et = lambda t: Edc * (t >= self.t_min and t <= self.t_max)


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
        self.Et = lambda t: E0 * np.pow(np.sin(omega0 * (t - t_shift) / nc / 2), 2) * np.cos(omega0 * (t - t_shift) + phi0) * (t >= self.t_min and t <= self.t_max)


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
        self.At = lambda t: A0 * np.pow(np.sin(omega0 * (t - t_shift) / nc / 2), 2) * np.sin(omega0 * (t - t_shift) + phi0) * (t >= self.t_min and t <= self.t_max)

    def get_At_data(self):
        ts = self.get_ts()
        vectorized_At = np.vectorize(self.At)
        return vectorized_At(ts)
    
    def get_Et_data(self):
        return -np.gradient(self.get_At_data(), self.delta_t)



###########################################3

def tdse_fd1d_tsurf(world, buffer, wave, light_field, Xi, logging_interval=500):
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
        if i % logging_interval == 0:
            energy = get_energy_1d(buffer, wave)
            norm = get_norm_1d(world, wave)
            print(f"[TDSE-fd1d] energy = {energy}, norm = {norm}")

    tsurf_res = [X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals]
    return tsurf_res


def tdse_fd1d_hg(world, buffer, wave, light_field, logging_interval=500):
    ts = light_field.get_ts()
    At_data = light_field.get_At_data()
    Et_data = light_field.get_Et_data()
    accel_expect_data = np.zeros(len(ts))
    pos_expect_data = np.zeros(len(ts))

    for (i, t) in enumerate(ts):
        tdse_laser_fd1d_onestep(buffer, wave, At=At_data[i])

        accel_expect_data[i] = get_accel_expect_1d(world=world, wavefunc=wave) - Et_data[i]
        pos_expect_data[i] = get_pos_expect_1d(world=world, wavefunc=wave)

        # logging
        if i % logging_interval == 0:
            energy = get_energy_1d(buffer, wave)
            norm = get_norm_1d(world, wave)
            print(f"[TDSE-fd1d] energy = {energy}, norm = {norm}")

    return [accel_expect_data, pos_expect_data]


def tdse_fd1d_hg_tsurf(world, buffer, wave, light_field, Xi, logging_interval=500):
    ts = light_field.get_ts()
    At_data = light_field.get_At_data()
    Et_data = light_field.get_Et_data()
    accel_expect_data = np.zeros(len(ts))
    pos_expect_data = np.zeros(len(ts))
    X_pos_vals = np.zeros(len(ts), dtype=complex)
    X_neg_vals = np.zeros(len(ts), dtype=complex)
    X_pos_dvals = np.zeros(len(ts), dtype=complex)
    X_neg_dvals = np.zeros(len(ts), dtype=complex)

    for (i, t) in enumerate(ts):
        tdse_laser_fd1d_onestep(buffer, wave, At=At_data[i])

        accel_expect_data[i] = get_accel_expect_1d(world=world, wavefunc=wave) - Et_data[i]
        pos_expect_data[i] = get_pos_expect_1d(world=world, wavefunc=wave)

        X_pos_vals[i] = (get_wave_value_1d(world, wave, Xi))
        X_neg_vals[i] = (get_wave_value_1d(world, wave, -Xi))
        X_pos_dvals[i] = (get_wave_1diff_value_1d(world, wave, Xi))
        X_neg_dvals[i] = (get_wave_1diff_value_1d(world, wave, -Xi))

        # logging
        if i % logging_interval == 0:
            energy = get_energy_1d(buffer, wave)
            norm = get_norm_1d(world, wave)
            print(f"[TDSE-fd1d] energy = {energy}, norm = {norm}")

    tsurf_res = [X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals]
    return [accel_expect_data, pos_expect_data, tsurf_res]


def tsurf_1d(tsurf_res, light_field, k_min, k_max, Xi, sampling_num=500):
    print(f"[TSURF-1d] Excecuting t-surf(1d) ...")
    X_pos_vals, X_neg_vals, X_pos_dvals, X_neg_dvals = tsurf_res
    ts = light_field.get_ts()
    At_data = light_field.get_At_data()
    ks = np.linspace(k_min, k_max, num=sampling_num)
    b1k = np.zeros(len(ks), dtype=complex)
    b2k = np.zeros(len(ks), dtype=complex)
    alpha = np.cumsum(At_data) * light_field.delta_t
    hanning_window = lambda t: 0.5 * (1.0 - np.cos(2 * np.pi * t / ts[-1]))

    foo1 = np.zeros(len(ts), dtype=complex)
    foo2 = np.zeros(len(ts), dtype=complex)
    
    # for (i, k) in enumerate(ks):
    #     for (j, t) in enumerate(ts):
    #         b1k[i] += hanning_window(t, ts[-1]) * (light_field.delta_t / np.sqrt(2 * np.pi)) * np.exp(1.0j * t * k * k / 2) * np.exp(-1.0j * k * (Xi - alpha[j])) * ((0.5 * k + At_data[j]) * X_pos_vals[j] - 0.5j * X_pos_dvals[j])
    #         b2k[i] += hanning_window(t, ts[-1]) * (-light_field.delta_t / np.sqrt(2 * np.pi)) * np.exp(1.0j * t * k * k / 2) * np.exp(-1.0j * k * (-Xi - alpha[j])) * ((0.5 * k + At_data[j]) * X_neg_vals[j] - 0.5j * X_neg_dvals[j])

    for (i, k) in enumerate(ks):
        foo1 = hanning_window(ts) * (light_field.delta_t / np.sqrt(2 * np.pi)) * np.exp(1.0j * ts * (k * k / 2)) * np.exp(-1.0j * k * (Xi - alpha)) * ((0.5 * k + At_data) * X_pos_vals - 0.5j * X_pos_dvals)
        foo2 = hanning_window(ts) * (-light_field.delta_t / np.sqrt(2 * np.pi)) * np.exp(1.0j * ts * (k * k / 2)) * np.exp(-1.0j * k * (-Xi - alpha)) * ((0.5 * k + At_data) * X_neg_vals - 0.5j * X_neg_dvals)
        b1k[i] = np.sum(foo1)
        b2k[i] = np.sum(foo2)

    Pk = np.abs(b1k + b2k) ** 2
    return ks, Pk


def fft_phy(ft_data, delta_t):
    N = len(ft_data)
    windows_data = chebwin(N, at=100)
    # windows_data = np.hanning(N)

    # rate = 1.0
    # a = math.floor(N * rate)
    # b = math.floor(N * (1 - rate) / 2)
    # windows_data = np.hanning(a)
    # windows_data = np.pad(windows_data, (b, N - a - b), mode='constant', constant_values=0)

    gk_data = np.fft.fft(ft_data * windows_data) * delta_t / np.sqrt(2 * np.pi)
    ks = np.fft.fftfreq(N, delta_t) * (2 * np.pi)
    delta_k = ks[1] - ks[0]
    return gk_data, ks, delta_k


def get_hg_spectrum_1d(ts, accel_data, pos_data, max_k):
    delta_t = ts[1] - ts[0]
    N = len(accel_data)
    accel_fft_data, ks, delta_k = fft_phy(accel_data, delta_t)
    pos_fft_data, _, _ = fft_phy(pos_data, delta_t)

    hg1 = (1.0 / ks) * np.pow(np.abs(accel_fft_data), 2)
    hg2 = np.pow(ks, 3) * np.pow(np.abs(pos_fft_data), 2)

    max_id = math.floor(max_k / delta_k)
    return hg1[0: max_id], hg2[0: max_id], ks[0: max_id]


def display_time_SI(t):
    res = t * 0.02418884326585
    print(f"{res:.3f} fs")
    return res

def display_ang_freq_SI(omega):
    freq = omega / (2 * np.pi)
    res = freq * 4.134137333 * 1e4
    print(f"{res:.3f} THz")
    return res

def display_electric_field_SI(E):
    res = E * 5.142206747 * 1e6
    print(f"{res:.3f} kV/cm")
    return res


import re

def EXPORT(var, name):
    pass    # Do nothing, just an idenfier.
    return var

def replace_export(match):
    str1 = match.group(1).strip()
    str2 = match.group(2).strip().strip("'\"")
    return f'( globals()["{str2}"] if ("{str2}" in globals()) else ({str1}) )'

def transform_code_string(input_str):
    pattern = r'EXPORT\s*\(\s*([^\s,]+)\s*,\s*([^\)]+)\s*\)'
    result = re.sub(pattern, replace_export, input_str)
    return result

def execute_code(script_name, global_vars_dict):
    code = open(script_name, "r", encoding="utf-8").read()
    exec(transform_code_string(code), global_vars_dict)