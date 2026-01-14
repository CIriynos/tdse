from tdse import *
from multiprocessing import Pool
import re

boundary_condition = "period"
execute_code("./test3.py", globals())
# eigen_waves = load_complex_arrays_from_hdf5("eigen_waves.h5")
eigen_waves_free = load_complex_arrays_from_hdf5("eigen_waves_free.h5")
eigen_waves = eigen_waves_free[0: 50]

en_list = []
p_list = []

fig, ax = plt.subplots(2, 1)
for i in range(0, len(eigen_waves)):
    en = py_get_energy_1d(rt, eigen_waves[i])
    # psi_pos, psi_neg, k, psi_k = separate_momentum_components(eigen_waves[i], xgrid)
    # p1 = py_get_kinetic_momentum_1d(rt, psi_pos)
    en_list.append(en)
    # p_list.append(p1)
ax[0].plot(np.real(eigen_waves[-1]))
ax[1].plot(en_list)

# plt.show()

def py_get_dipole_transition_1d(rt, wave1, wave2):
    wave1 /= np.sqrt(np.vdot(wave1, wave1))
    wave2 /= np.sqrt(np.vdot(wave2, wave2))
    wave1 *= np.hanning(len(wave1))
    wave2 *= np.hanning(len(wave2))
    xgrid = create_grid_data(rt["Nx"], rt["delta_x"], rt["shift_x"])
    dipole_transition = np.vdot(wave1, xgrid * wave2)
    return dipole_transition

dipole_transitions_matrix = np.zeros((len(eigen_waves), len(eigen_waves)), dtype=complex)
for i in range(0, len(eigen_waves)):
    for j in range(0, len(eigen_waves)):
        dipole_transitions_matrix[i, j] = py_get_dipole_transition_1d(rt, eigen_waves[i], eigen_waves[j])

fig, ax = plt.subplots()

ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.5), cmap='hot')
ax.set_title("Dipole Transition Matrix Magnitude")
ax.set_xlabel("Eigenstates")
ax.set_ylabel("Eigenstates")
ax.set_xticks(range(0, len(eigen_waves), 5))
ax.set_yticks(range(0, len(eigen_waves), 5))
ax.set_xticklabels([f"{en_list[i]:.2f}" for i in range(0, len(en_list), 5)])
ax.set_yticklabels([f"{en_list[i]:.2f}" for i in range(0, len(en_list), 5)])
fig.colorbar(ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.5), cmap='hot'), ax=ax, label='Magnitude')

print(dipole_transitions_matrix[0, 1])
print(dipole_transitions_matrix[-1, -2])

plt.show()  