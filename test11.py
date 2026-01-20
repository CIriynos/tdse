from tdse import *
from multiprocessing import Pool
import re

boundary_condition = "period"
execute_code("./test3.py", globals())

eigen_waves = load_complex_arrays_from_hdf5("eigen_waves.h5")
eigen_waves_free = load_complex_arrays_from_hdf5("eigen_waves_free.h5")
eigen_waves_all = eigen_waves + eigen_waves_free[::1]
dipole_transitions_matrix = load_complex_matrix_from_hdf5("dipole_transitions_matrix.h5")

en_list = []
for i in range(0, len(eigen_waves_all)):
    en = py_get_energy_1d(rt, eigen_waves_all[i])
    en_list.append(en)

#################

# fig, ax = plt.subplots()
# ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.5), cmap='hot')
# ax.set_title("Dipole Transition Matrix Magnitude")
# ax.set_xlabel("Eigenstates")
# ax.set_ylabel("Eigenstates")
# ax.set_xticks(range(0, len(eigen_waves_all), 20))
# ax.set_yticks(range(0, len(eigen_waves_all), 20))
# ax.set_xticklabels([f"{en_list[i]:.1f}" for i in range(0, len(en_list), 20)])
# ax.set_yticklabels([f"{en_list[i]:.1f}" for i in range(0, len(en_list), 20)])
# fig.colorbar(ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.5), cmap='hot'), ax=ax, label='Magnitude')

# fig, ax = plt.subplots()
# ax.plot(np.abs(dipole_transitions_matrix.diagonal()))
# plt.show()

#################
from scipy import sparse
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded


bound_state_indices = [i for i in range(len(eigen_waves) - 42)]
D = dipole_transitions_matrix[np.ix_(bound_state_indices, bound_state_indices)]

fig, ax = plt.subplots()
ax.imshow(np.pow(np.abs(D), 0.25), cmap='hot')

def create_T_matrix(N, t):
    T = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            T[i, j] = np.exp(1j * (en_list[i] - en_list[j]) * t)
    return T

def create_t_vec(N, t):
    t_vec = np.zeros(N, dtype=complex)
    for i in range(N):
        t_vec[i] = np.exp(-1j * en_list[i] * t)
    return t_vec

# propagation
N = len(bound_state_indices)
delta_t = 0.05

E0 = 0.03
omega0 = 0.057
nc = 15
Edc = 0
laser = cos2_laser_pulse(delta_t=delta_t, E0=E0, omega0=omega0, nc=nc)
dc_bias = dc_bias(delta_t, Edc, laser.get_duration())
laser_all = combine_light_field(laser, dc_bias)

Et_data = laser_all.get_Et_data()
steps = laser_all.get_steps()

Dt = D.copy()
dipole_values = np.zeros(steps, dtype=complex)
state = np.zeros(N, dtype=complex)
state[0] = 1.0 + 0.0j
absorb_mask = np.ones(N, dtype=float)
absorb_mask[-1] = 0.0
absorb_mask[-2] = 0.5
absorb_mask[-3] = 0.8
absorb_mask[-4] = 0.9
absorb_mask[-5] = 0.98

for i in range(0, steps):
    t = i * delta_t
    T = create_T_matrix(N, t)
    Dt = D * Et_data[i] * T
    A = np.eye(N, dtype=complex) + 1j * delta_t * Dt / 2.0
    B = np.eye(N, dtype=complex) - 1j * delta_t * Dt / 2.0
    state = np.linalg.solve(A, B @ state)
    state_t = state * create_t_vec(N, t)
    dipole_values[i] = np.vdot(state_t, D @ state_t)
    state = state * absorb_mask
    if i % 200 == 0:
        print("step:", i, " time:", t, " |state|:", np.linalg.norm(state))

fig, ax = plt.subplots()
ax.bar(range(N), np.abs(state)**2)
ax.set_yscale('log')

fig, ax = plt.subplots()
hg1_py, hg2_py, ks_py = get_hg_spectrum_1d(laser.get_ts(), dipole_values, dipole_values, max_k=(30) * omega0)
ax.plot(ks_py / laser.omega0, hg2_py, label="pos")
plt.xticks(range(1, (30), 2))
plt.grid(True, alpha=0.2)
plt.yscale('log')
plt.ylim(1e-15, 1e2)
plt.legend()

# fig, ax = plt.subplots()
# ax.plot(np.real(dipole_values))
# ax.plot(np.imag(dipole_values))

# plt.show()

#################

rate = 1
Nx = 3600 * rate
delta_x = 0.2 / rate
delta_t = 0.05 / rate
Xi = Nx * delta_x / 2 * 0.8
execute_code("./test.py", globals())

fig, ax = plt.subplots()
ax.plot(ks_py / laser.omega0, hg2_py, label="pos")
ax.plot(ks / laser.omega0, hg2, label="pos_tdse", linestyle='dashed')
plt.xticks(range(1, (30), 2))
plt.grid(True, alpha=0.2)
plt.yscale('log')
plt.ylim(1e-15, 1e2)
plt.legend()


plt.show()