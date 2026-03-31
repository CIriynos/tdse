# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tdse import *

# Grid
Nx =            EXPORT(5000 * 1, "Nx")
delta_x =       EXPORT(0.2 / 1, "delta_x")
shift_x =       -(Nx * delta_x) / 2
Lx =            Nx * delta_x

# Time evolution
delta_t =       EXPORT(0.05 / 1, "delta_t")
imag_delta_t =  EXPORT(0.1, "imag_delta_t")
itp_steps =     EXPORT(1000, "itp_steps")

# t-surf
Xi =            EXPORT((Lx * 0.5 * 0.8), "Xi")

# laser
E0 =            EXPORT(0.03, "E0")
omega0 =        EXPORT(0.057 * 0.5, "omega0")
nc =            EXPORT(6, "nc")
Edc =           EXPORT(0.0, "Edc")
laser = cos2_laser_pulse(delta_t=delta_t, E0=E0, omega0=omega0, nc=nc)
dc_bias_ = dc_bias(delta_t, Edc, laser.get_duration())
laser_all = combine_light_field(laser, dc_bias_)

# potential 
a0 =            EXPORT(2.0, "a0")
b0 =            EXPORT(1.0, "b0")
short_range_flag =    EXPORT(True, "short_range_flag")
short_range_sigma =   EXPORT(20.0, "short_range_sigma")
Vx =            lambda x: -b0 / np.sqrt(x * x + a0) * (np.exp(- x * x / (short_range_sigma ** 2)) if short_range_flag == True else 1.0)
Vx_absorb =     lambda x: -100j * pow((np.abs(x) - Xi) / (Lx / 2 - Xi), 8) * (np.abs(x) > Xi)


# enviroment
world, xs = create_physical_world_1d(Nx, delta_x, shift_x, Vx, Vx_absorb)
buffer = create_runtime_buffer_1d(world, delta_t, imag_delta_t)
wave = get_ground_state_1d(buffer, itp_steps)
init_energy = get_energy_1d(buffer, wave)

# get eigenstates
# ek_list = [-0.669, -0.27, -0.15, -0.09, -0.063, -0.045, -0.034, -0.026, -0.0217, -0.017,
#     -0.0148, -0.0125, -0.0108, -0.0093, -0.0082, -0.0072, -0.0064, -0.0057, -0.0052, -0.0047,
#     -0.0042, -0.0039, -0.0036, -0.0033, -0.0030, -0.0028, -0.0026, -0.0024, -0.0022, -0.0021,
#     -0.0020, -0.00187, -0.00177, -0.00166, -0.00157, -0.00149, -0.00141, -0.00134, -0.00127, -0.00121]    # for a0 = 1.0

# for a0 = 2.0 (Ip = 0.5)
# ek_list = [np.float64(-0.500000021292431), np.float64(-0.23290336161288572), np.float64(-0.13382887730918563), np.float64(-0.08477791602413531), np.float64(-0.05885737800031285), np.float64(-0.04281863155666199), np.float64(-0.03274088654914872), np.float64(-0.025681385480090826), np.float64(-0.020784831616617115), np.float64(-0.0171285), np.float64(-0.04281863155666199), np.float64(-0.03274088654914872), np.float64(-0.025681385480090826), np.float64(-0.020784831616617115), np.float64(-0.01708357667206825), np.float64(-0.014349273718118302), np.float64(-0.012174360991161261), np.float64(-0.010496151960642172), np.float64(-0.009111766118164479), np.float64(-0.00800901241752417), np.float64(-0.007074168068594782), np.float64(-0.0063112246459950064), np.float64(-0.005650596343793669), np.float64(-0.005101054326910255), np.float64(-0.004617083535377157), np.float64(-0.004208239255517555), np.float64(-0.0038431616527085845), np.float64(-0.0035308105569542407), np.float64(-0.003248665600247292), np.float64(-0.0030046835909982124), np.float64(-0.002782139865819993), np.float64(-0.0025879454235535455), np.float64(-0.0024093314109559904), np.float64(-0.0022522497820749478), np.float64(-0.002106723960673608), np.float64(-0.0019778700370101125), np.float64(-0.0018577392309129638), np.float64(-0.0017507353111119524), np.float64(-0.001650418330623477), np.float64(-0.0015605907737801003), np.float64(-0.001475959175799901), np.float64(-0.0013998199820029185), np.float64(-0.0013277669341230825), np.float64(-0.0012626703752922983), np.float64(-0.0012008217044670118), np.float64(-0.0011447315871357417), np.float64(-0.0010912478523336243), np.float64(-0.0010425764812443449), np.float64(-0.0009960147211913928), np.float64(-0.0009535093906594015), np.float64(-0.0009127248391977993), np.float64(-0.0008753864741925319), np.float64(-0.0008394615654453039), np.float64(-0.0008064855129563723), np.float64(-0.0007746778489583714), np.float64(-0.000745410187282477), np.float64(-0.0007171138936222379), np.float64(-0.0006910187786367547)]

# for a0 = 2.0 with short range sigma = 5.0
# ek_list = [np.float64(-0.4752654358081353), np.float64(-0.15756026030786796), np.float64(-0.028532769561578065)]

# for a0 = 2.0 with short range sigma = 2.5
# ek_list = [np.float64(-0.42355649852346483), np.float64(-0.049503848810466544)]

# for sigma = 20.0
ek_list = [np.float64(-0.4982364180407881), np.float64(-0.22654507871076526), np.float64(-0.12180292994831853), np.float64(-0.06662987305157551), np.float64(-0.035328050634033145), np.float64(-0.015446590924057098), np.float64(-0.004159471981263993)]

eigen_waves = []
for i in range(0, len(ek_list)):
    ek = ek_list[i]
    imag_delta_t_tmp = get_imag_delta_t_from_ek(ek)
    buffer_tmp = create_runtime_buffer_1d(world, delta_t, imag_delta_t_tmp)
    e_wave = get_ground_state_1d(buffer_tmp, itp_steps)
    eigen_waves.append(e_wave)

# tdse-hg
accel, pos, accel_free, pos_free, accel_bound, pos_bound, pos_cross, bound_norm_data = tdse_fd1d_hg_analytical(world, buffer, wave, light_field=laser_all, Xi=Xi, bound_states=eigen_waves)

plt.figure()
plt.plot(laser.get_ts(), pos, label=" expect pos total")
plt.plot(laser.get_ts(), pos_free, label=" expect pos free")
plt.plot(laser.get_ts(), pos_bound, label=" expect pos bound")
plt.plot(laser.get_ts(), pos_cross, label=" expect pos cross")
plt.legend()

# plt.plot(laser.get_ts(), bound_norm_data, label="bound norm")
# plt.legend()

# plt.figure()
# plt.plot(laser.get_ts(), accel, label="expect accel total")
# plt.plot(laser.get_ts(), accel_free, label="expect accel free")
# plt.plot(laser.get_ts(), accel_bound, label="expect accel bound")
# plt.legend()

# harmonic spectrum
n_cut_off_estim = math.floor((-init_energy + 3.17 * (E0 ** 2.0 / (4.0 * (omega0 ** 2.0)))) / omega0)
print(f"n_cut_off = {n_cut_off_estim}")
max_display_n = EXPORT(30, "max_display_n")
display_spacing_n = EXPORT(2, "display_spacing_n")
display_start_n = EXPORT(1, "display_start_n")

# pos
pos_expect_spectrum, ks = get_pos_expect_spectrum_1d(laser.get_ts(), pos, max_k= max_display_n * omega0)
pos_expect_spectrum_free, _ = get_pos_expect_spectrum_1d(laser.get_ts(), pos_free, max_k= max_display_n * omega0)
pos_expect_spectrum_bound, _ = get_pos_expect_spectrum_1d(laser.get_ts(), pos_bound, max_k= max_display_n * omega0)
pos_expect_spectrum_residual = pos_expect_spectrum - pos_expect_spectrum_free - pos_expect_spectrum_bound
pos_expect_spectrum_cross, _ = get_pos_expect_spectrum_1d(laser.get_ts(), pos_cross, max_k= max_display_n * omega0)

plt.figure()
plt.plot(ks / omega0, np.pow(ks, 3) * np.pow(np.abs(pos_expect_spectrum), 2), label="pos expect total", lw = 2.0)
plt.plot(ks / omega0, np.pow(ks, 3) * np.pow(np.abs(pos_expect_spectrum_free), 2), label="pos expect free", lw = 1.5)
plt.plot(ks / omega0, np.pow(ks, 3) * np.pow(np.abs(pos_expect_spectrum_bound), 2), label="pos expect bound", lw = 1.5)
plt.plot(ks / omega0, np.pow(ks, 3) * np.pow(np.abs(pos_expect_spectrum_residual), 2), label="pos expect residual", lw = 1.5)
plt.plot(ks / omega0, np.pow(ks, 3) * np.pow(np.abs(pos_expect_spectrum_cross), 2), label="pos expect cross", lw = 1.5)
plt.plot(ks / omega0, np.pow(ks, 3) * np.pow(np.abs(pos_expect_spectrum_cross + pos_expect_spectrum_bound + pos_expect_spectrum_free), 2), label="pos expect recover", lw = 1.5)

plt.xticks(range(display_start_n, max_display_n, display_spacing_n))
plt.grid(True, alpha=0.2)
plt.yscale('log')
plt.ylim(1e-12, 1e3)
plt.legend()
plt.xlabel("Harmonic order (n)")
plt.ylabel("Harmonic spectrum amplitude (arb. unit)")

plt.show()



# # accel
# pos_expect_spectrum, ks = get_pos_expect_spectrum_1d(laser.get_ts(), accel, max_k= max_display_n * omega0)
# pos_expect_spectrum_free, _ = get_pos_expect_spectrum_1d(laser.get_ts(), accel_free, max_k= max_display_n * omega0)
# pos_expect_spectrum_bound, _ = get_pos_expect_spectrum_1d(laser.get_ts(), accel_bound, max_k= max_display_n * omega0)
# pos_expect_spectrum_residual = pos_expect_spectrum - pos_expect_spectrum_free - pos_expect_spectrum_bound

# plt.figure()
# plt.plot(ks / omega0, (1.0 / ks) * np.pow(np.abs(pos_expect_spectrum), 2), label="accel expect total", lw = 2.0)
# plt.plot(ks / omega0, (1.0 / ks) * np.pow(np.abs(pos_expect_spectrum_free), 2), label="accel expect free", lw = 1.5)
# plt.plot(ks / omega0, (1.0 / ks) * np.pow(np.abs(pos_expect_spectrum_bound), 2), label="accel expect bound", lw = 1.5)
# # plt.plot(ks / omega0, (1.0 / ks) * np.pow(np.abs(pos_expect_spectrum_residual), 2), label="accel expect residual", lw = 1.5)

# plt.xticks(range(display_start_n, max_display_n, display_spacing_n))
# plt.grid(True, alpha=0.2)
# plt.yscale('log')
# plt.ylim(1e-12, 1e3)
# plt.legend()
# plt.xlabel("Harmonic order (n)")
# plt.ylabel("Harmonic spectrum amplitude (arb. unit)")

# plt.show()