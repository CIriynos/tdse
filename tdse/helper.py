# -*- coding: utf-8 -*-
import math
import os
import numpy as np
from ctypes import *
from ctypes import CDLL
import matplotlib.pyplot as plt
import sys
from scipy.signal.windows import chebwin
import h5py

if os.name == "nt":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib = CDLL(current_dir + r"\interface.dll")
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


def create_grid_data(N, delta, shift):
    return np.array([shift + delta * i for i in range(0, N)])

def create_physical_world_1d(
    Nx, delta_x, shift_x,   # x-grid
    potential,
    absorption_potential
    ):

    xs = create_grid_data(Nx, delta_x, shift_x)
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

def get_wave_value_list_1d(world, wavefunc, Nx):
    c_array_t = c_double * Nx
    wave_real = c_array_t()
    wave_imag = c_array_t()
    
    lib.get_wave_value_list_1d(c_void_p(world), c_void_p(wavefunc), wave_real, wave_imag)
    
    value_list = np.zeros(Nx, dtype=complex)
    for i in range(0, Nx):
        value_list[i] = wave_real[i] + wave_imag[i] * (1.0j)

    return value_list

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

####################################

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

####################################

import re

def EXPORT(var, name):
    pass    # Do nothing, just an idenfier.
    return var

def replace_export_pattern(text):
    """
    将字符串中的 EXPORT(str1,str2) 替换为指定格式
    """
    # 正则表达式模式
    # EXPORT\s*\(\s* 匹配 EXPORT( 及其前后的空格
    # (.*?) 匹配第一个参数（非贪婪模式，保留内部空格）
    # \s*,\s* 匹配逗号及其前后的空格
    # (.*?) 匹配第二个参数（非贪婪模式）
    # \s*\) 匹配右括号及其前面的空格
    pattern = r'EXPORT\s*\(\s*(.*?)\s*,\s*(.*?)\s*\)'
    
    def replacement(match):
        # 提取两个参数
        str1 = match.group(1).strip()  # 去除str1的前后空格，保留内部空格
        str2_raw = match.group(2).strip()  # 去除str2的前后空格
        
        # 移除str2中的所有单引号和双引号
        str2 = str2_raw.replace('"', '').replace("'", '')
        
        # 构建替换字符串
        return f'( globals()["{str2}"] if ("{str2}" in globals() ) else ({str1}) )'
    
    # 使用re.sub进行替换
    return re.sub(pattern, replacement, text, flags=re.DOTALL)

def execute_code(script_name, global_vars_dict):
    code = open(script_name, "r", encoding="utf-8").read()
    # print(replace_export_pattern(code))
    exec(replace_export_pattern(code), global_vars_dict)


###################################
import numpy as np
import h5py
import os
from pathlib import Path

def save_complex_arrays_to_hdf5(arrays, filename, compression=None):
    
    # 确保目录存在
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 定义复数的复合数据类型
    complex_dtype = np.dtype([('real', np.float64), ('imag', np.float64)])
    
    # 写入HDF5文件
    with h5py.File(filename, 'w') as f:
        # 存储数组数量作为元数据
        f.attrs['num_arrays'] = len(arrays)
        
        for i, arr in enumerate(arrays):
            if not np.issubdtype(arr.dtype, np.complexfloating):
                raise ValueError(f"数组 {i} 不是复数类型")
            
            # 转换为结构化数组
            structured_arr = np.empty(arr.shape, dtype=complex_dtype)
            structured_arr['real'] = arr.real
            structured_arr['imag'] = arr.imag
            
            # 创建数据集
            dataset_name = f'array_{i}'
            if compression:
                f.create_dataset(dataset_name, data=structured_arr, 
                               compression=compression)
            else:
                f.create_dataset(dataset_name, data=structured_arr)


def load_complex_arrays_from_hdf5(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"HDF5文件不存在: {filename}")
    
    loaded_arrays = []
    
    with h5py.File(filename, 'r') as f:
        # 获取数组数量（如果有存储的话）
        if 'num_arrays' in f.attrs:
            num_arrays = f.attrs['num_arrays']
            # 按顺序读取
            for i in range(num_arrays):
                dataset_name = f'array_{i}'
                if dataset_name not in f:
                    raise ValueError(f"数据集 {dataset_name} 不存在")
                
                structured = f[dataset_name][()]
                complex_arr = structured['real'] + 1j * structured['imag']
                loaded_arrays.append(complex_arr)
        else:
            # 如果没有num_arrays属性，按字典序读取所有array_*数据集
            array_keys = [key for key in f.keys() if key.startswith('array_')]
            # 按数字顺序排序
            array_keys.sort(key=lambda x: int(x.split('_')[1]))
            
            for key in array_keys:
                structured = f[key][()]
                complex_arr = structured['real'] + 1j * structured['imag']
                loaded_arrays.append(complex_arr)
    
    return loaded_arrays


####################################
# TDSE
from scipy import sparse
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

def get_time(f):
    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

def create_sparse_dia_mat(N, diag_num_list):
    band_width = len(diag_num_list)
    shift = band_width // 2
    identifier_list = [i for i in range(0 - shift, band_width - shift)]
    data = [np.ones(N - abs(i - shift)) * diag_num_list[i] for i in range(0, band_width)]
    return sparse.diags(data, identifier_list, format='dia', dtype="complex")

def create_band(A_dia):
    N = A_dia.shape[0]
    data = A_dia.data
    offsets = A_dia.offsets
    l = -offsets.min()
    u = offsets.max()

    ab = np.zeros((l + u + 1, N), dtype="complex")
    for i, offset in enumerate(offsets):
        row = u - offset
        ab[row] = data[i]
    return {"ab": ab, "l": l, "u": u}

def py_thomas_solve(A, b):
    solve_banded((A["l"], A["u"]), A["ab"], b, overwrite_b=True)

def py_create_buffer_1d(Nx, delta_x, delta_t, delta_t_itp, shift_x,
    Vx, Vx_absorb, accuracy=4, boundary_condition="reflect"):
    
    xgrid = create_grid_data(Nx, delta_x, shift_x)
    I = create_sparse_dia_mat(Nx, [1])
    V = create_sparse_dia_mat(Nx, [Vx(xgrid)])
    V_absorb = create_sparse_dia_mat(Nx, [Vx(xgrid) + Vx_absorb(xgrid)])

    if accuracy == 4:
        D2 = create_sparse_dia_mat(Nx, [-1, 16, -30, 16, -1]) * (1.0 / (12.0 * delta_x * delta_x))
        D1 = create_sparse_dia_mat(Nx, [1, -8, 0, 8, -1]) * (1.0 / (12.0 * delta_x))
        
        M = I   # meaningless
        N = I   # meaningless
        A_pos = I + (0.25 * 1j * delta_t) * D2 + (-0.5 * 1j * delta_t) * V_absorb
        A_neg = I + (-0.25 * 1j * delta_t) * D2 + (0.5 * 1j * delta_t) * V_absorb
        A_pos_itp = I + (0.25 * delta_t_itp) * D2 + (-0.5 * delta_t_itp) * V
        A_neg_itp = I + (-0.25 * delta_t_itp) * D2 + (0.5 * delta_t_itp) * V

        H = -0.5 * D2 + V
        H_absorb = -0.5 * D2 + V_absorb

        A_neg_ab = create_band(A_neg.todia())
        A_pos_csr = A_pos.tocsr()
        A_neg_itp_ab = create_band(A_neg_itp.todia())
        A_pos_itp_csr = A_pos_itp.tocsr()

    elif accuracy == 2:
        if boundary_condition == "reflect":
            D2 = create_sparse_dia_mat(Nx, [1, -2, 1]) * (1.0 / (delta_x * delta_x))
            D1 = create_sparse_dia_mat(Nx, [-1, 0, 1]) * (1.0 / (2 * delta_x))        

            M = I + (delta_x * delta_x / 12) * D2
            N = I + (delta_x * delta_x / 6) * D2
            A_pos = M - (D2 * (-0.5) + M * V_absorb) * (0.5j * delta_t)
            A_neg = M + (D2 * (-0.5) + M * V_absorb) * (0.5j * delta_t)
            A_pos_itp = M - (D2 * (-0.5) + M * V) * (0.5j * (-1j * delta_t_itp))
            A_neg_itp = M + (D2 * (-0.5) + M * V) * (0.5j * (-1j * delta_t_itp))

            A_neg_itp_ab = create_band(A_neg_itp.todia())
            A_pos_itp_csr = A_pos_itp.tocsr()
            A_neg_ab = create_band(A_neg.todia())
            A_pos_csr = A_pos.tocsr()
        
        elif boundary_condition == "period":
            D2 = create_sparse_dia_mat(Nx, [1, -2, 1]).tocsr() * (1.0 / (delta_x * delta_x))
            D1 = create_sparse_dia_mat(Nx, [-1, 0, 1]).tocsr() * (1.0 / (2 * delta_x))
            D2[0, Nx - 1] = (1.0 / (delta_x * delta_x))
            D2[Nx - 1, 0] = (1.0 / (delta_x * delta_x))
            D1[0, Nx - 1] = (-1.0 / (2 * delta_x))
            D1[Nx - 1, 0] = (1.0 / (2 * delta_x))

            M = I + (delta_x * delta_x / 12) * D2
            N = I + (delta_x * delta_x / 6) * D2
            A_pos = M - (D2 * (-0.5) + M * V_absorb) * (0.5j * delta_t)
            A_neg = M + (D2 * (-0.5) + M * V_absorb) * (0.5j * delta_t)
            A_pos_itp = M - (D2 * (-0.5) + M * V) * (0.5j * (-1j * delta_t_itp))
            A_neg_itp = M + (D2 * (-0.5) + M * V) * (0.5j * (-1j * delta_t_itp))
            P_pos_itp = N - D1 * (0.5 * delta_t_itp)
            P_neg_itp = N + D1 * (0.5 * delta_t_itp)

            # manually remove those elements
            A_neg_itp_tmp, a, b = prepare_for_rank1_update(A_neg_itp)
            A_neg_tmp, a, b = prepare_for_rank1_update(A_neg)
            P_neg_itp_tmp, _, _ = prepare_for_rank1_update(P_neg_itp)

            A_neg_itp_ab = create_band(A_neg_itp_tmp.todia())   # still keep origin
            A_neg_ab = create_band(A_neg_tmp.todia())   # still keep origin
            P_neg_itp_ab = create_band(P_neg_itp_tmp.todia())
            A_pos_itp_csr = A_pos_itp.tocsr() 
            A_pos_csr = A_pos.tocsr()
            P_pos_itp_csr = P_pos_itp.tocsr() 
        
        # meaningless in 2-order scenario
        H = -0.5 * D2 + V
        H_absorb = -0.5 * D2 + V_absorb

    result = {
        "Nx": Nx,
        "delta_x": delta_x,
        "delta_t": delta_t,
        "delta_t_itp": delta_t_itp,
        "shift_x": shift_x,
        
        "D2": D2,
        "D1": D1,
        "I": I,
        "V": V,
        "V_absorb": V_absorb,
        "M": M,
        "N": N,
        "A_pos": A_pos,
        "A_neg": A_neg,
        "A_pos_itp": A_pos_itp,
        "A_neg_itp": A_neg_itp,
        "H": H,
        "H_absorb": H_absorb,
        "A_neg_ab": A_neg_ab,
        "A_pos_csr": A_pos_csr,
        "A_neg_itp_ab": A_neg_itp_ab,
        "A_pos_itp_csr": A_pos_itp_csr,
        # "P_pos_itp": P_pos_itp,
        # "P_neg_itp": P_neg_itp,
        # "P_pos_itp_csr": P_pos_itp_csr,
        # "P_neg_itp_ab": P_neg_itp_ab,
        
        "accuracy": accuracy,
        "boundary_condition": boundary_condition
    }
    return result

def prepare_for_rank1_update(A):
    N = A.shape[0]
    A0 = A.copy()
    a = A0[0, N - 1]
    b = A0[N - 1, 0]
    A0[0, N - 1] = 0
    A0[N - 1, 0] = 0
    A0[0, 0] -= np.sqrt(a * b)
    A0[N - 1, N - 1] -= np.sqrt(a * b)
    A0.eliminate_zeros()
    return A0, a, b

def py_thomas_solve_rank1_update(A0, wave, a, b):
    N = A0["ab"].shape[1]
    u = np.zeros(N, dtype=complex)
    v = np.zeros(N, dtype=complex)
    x0 = wave.copy()
    py_thomas_solve(A0, x0)
    
    u[0] = np.sqrt(a)
    u[N - 1] = np.sqrt(b)
    v[0] = (np.sqrt(b)).conjugate()
    v[N - 1] = (np.sqrt(a)).conjugate()
    lam = u.copy()
    py_thomas_solve(A0, lam)

    x_prime = (u * np.vdot(v, x0)) / (np.vdot(v, lam) + 1)
    py_thomas_solve(A0, x_prime)
    x_final = x0 - x_prime
    wave[:] = x_final


gauss_pkg_f = lambda x, omega, k0, x0: (1.0 / np.pow(2 * np.pi, 0.25)) * np.exp(1j * k0 * x) * np.exp(-np.pow((x - x0) / (2 * omega), 2))

def py_itp_1d(rt, steps=1000):
    xgrid = create_grid_data(rt["Nx"], rt["delta_x"], rt["shift_x"])
    wave = gauss_pkg_f(xgrid, omega=1.0, k0=1.0, x0=1.0)

    for step in range(0, steps):
        wave[:] = rt["A_pos_itp_csr"] @ wave
        if rt["boundary_condition"] == "reflect":
            py_thomas_solve(rt["A_neg_itp_ab"], wave)
        elif rt["boundary_condition"] == "period":
            a = rt["A_neg_itp"][0, rt["Nx"] - 1]
            b = rt["A_neg_itp"][rt["Nx"] - 1, 0]
            py_thomas_solve_rank1_update(rt["A_neg_itp_ab"], wave, a, b)

        wave /= np.sqrt(np.vdot(wave, wave))
    return wave


def py_itp_free_1d(rt, steps=1000):
    xgrid = create_grid_data(rt["Nx"], rt["delta_x"], rt["shift_x"])
    wave = gauss_pkg_f(xgrid, omega=1.0, k0=1.0, x0=0.0)

    for step in range(0, steps):
        # wave[:] = rt["P_pos_itp_csr"] @ wave
        # if rt["boundary_condition"] == "reflect":
        #     py_thomas_solve(rt["P_neg_itp_ab"], wave)
        # elif rt["boundary_condition"] == "period":
        #     a = rt["P_neg_itp"][0, rt["Nx"] - 1]
        #     b = rt["P_neg_itp"][rt["Nx"] - 1, 0]
        #     py_thomas_solve_rank1_update(rt["P_neg_itp_ab"], wave, a, b)
        # wave /= np.sqrt(np.vdot(wave, wave))

        wave[:] = rt["A_pos_itp_csr"] @ wave
        if rt["boundary_condition"] == "reflect":
            py_thomas_solve(rt["A_neg_itp_ab"], wave)
        elif rt["boundary_condition"] == "period":
            a = rt["A_neg_itp"][0, rt["Nx"] - 1]
            b = rt["A_neg_itp"][rt["Nx"] - 1, 0]
            py_thomas_solve_rank1_update(rt["A_neg_itp_ab"], wave, a, b)
        wave /= np.sqrt(np.vdot(wave, wave))

    return wave


def tdse_fd1d(rt, wave, steps=1000):
    for step in range(0, steps):
        # print(np.vdot(wave, wave))
        wave[:] = rt["A_pos_csr"] @ wave
        if rt["boundary_condition"] == "reflect":
            py_thomas_solve(rt["A_neg_ab"], wave)
        elif rt["boundary_condition"] == "period":
            a = rt["A_neg"][0, rt["Nx"] - 1]
            b = rt["A_neg"][rt["Nx"] - 1, 0]
            py_thomas_solve_rank1_update(rt["A_neg_ab"], wave, a, b)
    return wave


def py_get_energy_1d(rt, wave):
    if rt["accuracy"] == 4:
        return np.real(np.vdot(wave, rt["H"] @ wave))
    elif rt["accuracy"] == 2:
        if rt["boundary_condition"] == "reflect":
            M_ab = create_band(rt["M"].todia())
            tmp1 = rt["D2"] @ wave
            py_thomas_solve(M_ab, tmp1)
            tmp2 = rt["V"] @ wave + (-0.5) * tmp1
            return np.real(np.vdot(wave, tmp2))

        elif rt["boundary_condition"] == "period":
            M_0, a, b = prepare_for_rank1_update(rt["M"])
            tmp1 = rt["D2"] @ wave
            py_thomas_solve_rank1_update(create_band(M_0.todia()), tmp1, a, b)
            tmp2 = rt["V"] @ wave + (-0.5) * tmp1
            return np.real(np.vdot(wave, tmp2))


def py_get_kinetic_momentum_1d(rt, wave):
    if rt["accuracy"] == 4:
        return np.real(np.vdot(wave, rt["D1"] @ wave) * (-1j))
    elif rt["accuracy"] == 2:
        if rt["boundary_condition"] == "reflect":
            N_ab = create_band(rt["N"].todia())
            tmp1 = rt["D1"] @ wave
            py_thomas_solve(N_ab, tmp1)
            return np.real(np.vdot(wave, tmp1) * (-1j))

        elif rt["boundary_condition"] == "period":
            N_0, a, b = prepare_for_rank1_update(rt["N"])
            tmp1 = rt["D1"] @ wave
            py_thomas_solve_rank1_update(create_band(N_0.todia()), tmp1, a, b)
            return np.real(np.vdot(wave, tmp1) * (-1j))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift


def separate_momentum_components(psi_x, x, plot=False):
    hbar=1.0

    # 1. 基本参数设置
    N = len(x)          # 网格点数
    L = x[-1] - x[0]    # 空间总长度
    dx = x[1] - x[0]    # 空间步长
    
    # 2. 验证均匀网格
    if not np.allclose(np.diff(x), dx, atol=1e-10):
        raise ValueError("位置网格必须是均匀的")
    
    # 3. FFT变换到动量空间
    psi_k = fft(psi_x) * dx / np.sqrt(2 * np.pi * hbar)  # 保持L2范数
    
    # 4. 生成物理动量网格 (p = ħk)
    # fftfreq返回的顺序: [0, 1, ..., N/2-1, -N/2, ..., -1] (N为偶数时)
    k = fftfreq(N, d=dx) * 2 * np.pi  # 波数 (rad/m)
    p = hbar * k                      # 物理动量
    
    # 5. 构建动量空间投影算符
    # 处理k=0点: 严格属于非正动量 (θ(0)=0)
    pos_mask = p > 0      # 正动量掩码 (p > 0)
    neg_mask = p <= 0     # 负动量掩码 (p <= 0)
    
    # 6. 应用投影
    psi_k_pos = np.zeros_like(psi_k)
    psi_k_neg = np.zeros_like(psi_k)
    psi_k_pos[pos_mask] = psi_k[pos_mask]
    psi_k_neg[neg_mask] = psi_k[neg_mask]
    
    # 7. 逆FFT回位置空间 (恢复范数)
    psi_pos = ifft(psi_k_pos) * np.sqrt(2 * np.pi * hbar) / dx
    psi_neg = ifft(psi_k_neg) * np.sqrt(2 * np.pi * hbar) / dx
    
    # 8. 归一化校正 (可选但推荐)
    # 保持分离后分量的相对范数
    norm_total = np.sqrt(np.trapz(np.abs(psi_x)**2, x))
    norm_pos = np.sqrt(np.trapz(np.abs(psi_pos)**2, x))
    norm_neg = np.sqrt(np.trapz(np.abs(psi_neg)**2, x))
    
    if norm_pos > 1e-10:
        psi_pos *= norm_total / norm_pos * np.sqrt(np.trapz(np.abs(psi_pos)**2, x)) / norm_total
    if norm_neg > 1e-10:
        psi_neg *= norm_total / norm_neg * np.sqrt(np.trapz(np.abs(psi_neg)**2, x)) / norm_total
    
    # 9. 验证 (可选)
    if plot:
        # 转换为物理单位的动量网格 (用于绘图)
        p_plot = fftshift(p)
        psi_k_plot = fftshift(np.abs(psi_k)**2)
        psi_k_pos_plot = fftshift(np.abs(psi_k_pos)**2)
        psi_k_neg_plot = fftshift(np.abs(psi_k_neg)**2)
        
        plt.figure(figsize=(12, 10))
        
        # 位置空间波函数
        plt.subplot(3, 1, 1)
        plt.plot(x, np.abs(psi_x)**2, 'k-', lw=2, label='Total |ψ|²')
        plt.plot(x, np.abs(psi_pos)**2, 'b--', lw=1.5, label='Positive momentum')
        plt.plot(x, np.abs(psi_neg)**2, 'r-.', lw=1.5, label='Negative momentum')
        plt.title('Position Space Wavefunction')
        plt.xlabel('Position (m)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        
        # 动量空间分布
        plt.subplot(3, 1, 2)
        plt.plot(p_plot, psi_k_plot, 'k-', lw=2, label='Total |ψ̃(p)|²')
        plt.fill_between(p_plot, 0, psi_k_pos_plot, color='blue', alpha=0.3, label='Positive momentum')
        plt.fill_between(p_plot, 0, psi_k_neg_plot, color='red', alpha=0.3, label='Negative momentum')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title('Momentum Space Distribution')
        plt.xlabel('Momentum (kg·m/s)')
        plt.ylabel('Spectral Density')
        plt.legend()
        plt.grid(True)
        
        # 验证投影性质
        plt.subplot(3, 1, 3)
        recon = psi_pos + psi_neg
        error = np.max(np.abs(recon - psi_x))
        plt.plot(x, np.abs(recon - psi_x), 'g-', lw=2)
        plt.title(f'Reconstruction Error (max|ψ_rec - ψ| = {error:.2e})')
        plt.xlabel('Position (m)')
        plt.ylabel('Absolute Error')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    psi_pos /= np.sqrt(np.vdot(psi_pos, psi_pos))
    psi_neg /= np.sqrt(np.vdot(psi_neg, psi_neg))

    return psi_pos, psi_neg, k, psi_k


#############


def find_all_points_with_error(a, b, query, delta, tol=1e-9):
    """
    在区间 (a, b) 内找出所有点，处理查询误差
    
    参数:
    a, b: 区间端点 (a < b)
    query: 查询函数，接收 x ∈ (a, b)，返回最近点（存在 < δ 的误差）
    delta: 最大查询误差（当 |p1-p2| <= 2*delta 时视为同一点）
    tol: 浮点容差（默认 1e-9），用于边界处理
    
    返回:
    list: 去重后的点集（每个簇取平均值作为代表）
    """
    if b - a < 2 * tol:
        raise ValueError("区间长度太小，无法进行有效查询")
    
    # 步骤1: 获取边界点
    left_val = query(a + tol)
    right_val = query(b - tol)
    
    # 情况1: 仅有一个点（观测距离 <= 2*delta）
    if abs(left_val - right_val) <= 2 * delta:
        return [left_val]
    
    # 收集所有原始点（含重复）
    all_points = [left_val, right_val]
    
    # 步骤2: 递归查找内部点（要求点间距 > 4*delta）
    def find_interior(L, R):
        """递归查找 (L, R) 内满足严格距离条件的点"""
        # 基本情况: 区间太小，不可能有新点（安全边际 4*delta）
        if R - L < 4 * delta:
            return []
        
        mid = (L + R) / 2.0
        y = query(mid)
        
        # 检查 y 是否是严格内部点（距离 L 和 R 均 > 2*delta）
        if L + 2 * delta < y < R - 2 * delta:
            # 递归搜索子区间
            left_pts = find_interior(L, y)
            right_pts = find_interior(y, R)
            return left_pts + [y] + right_pts
        return []
    
    # 获取内部点
    interior_points = find_interior(left_val, right_val)
    all_points.extend(interior_points)
    
    # 步骤3: 全局去重（合并同一物理点的多次测量）
    if not all_points:
        return []
    
    # 按值排序
    all_points.sort()
    
    # 合并距离 <= 2*delta 的点簇
    clusters = []
    current_cluster = [all_points[0]]
    
    for i in range(1, len(all_points)):
        # 检查当前点是否属于上一个簇
        if all_points[i] - current_cluster[-1] <= 2 * delta:
            current_cluster.append(all_points[i])
        else:
            # 完成当前簇，计算代表值
            cluster_avg = sum(current_cluster) / len(current_cluster)
            clusters.append(cluster_avg)
            current_cluster = [all_points[i]]
    
    # 处理最后一个簇
    if current_cluster:
        cluster_avg = sum(current_cluster) / len(current_cluster)
        clusters.append(cluster_avg)
    
    return clusters



##########################

def py_get_dipole_transition_1d(rt, wave1, wave2):
    wave1 /= np.sqrt(np.vdot(wave1, wave1))
    wave2 /= np.sqrt(np.vdot(wave2, wave2))
    xgrid = create_grid_data(rt["Nx"], rt["delta_x"], rt["shift_x"])
    dipole_transition = np.vdot(wave1, xgrid * wave2)
    return dipole_transition

def project_out(rt, wave, basis_waves):
    proj_wave = np.copy(wave)
    for bw in basis_waves:
        c = np.vdot(bw, proj_wave) / np.vdot(bw, bw)
        proj_wave -= c * bw
    proj_wave /= np.sqrt(np.vdot(proj_wave, proj_wave))
    return proj_wave

def get_coefficients(rt, wave, basis_waves):
    coeffs = []
    for bw in basis_waves:
        c = np.vdot(bw, wave) / np.vdot(bw, bw)
        coeffs.append(c)
    return coeffs

def reconstruct_wave(rt, coeffs, basis_waves):
    recon_wave = np.zeros_like(basis_waves[0])
    for i in range(0, len(basis_waves)):
        recon_wave += coeffs[i] * basis_waves[i]
    # recon_wave /= np.sqrt(np.vdot(recon_wave, recon_wave))
    return recon_wave