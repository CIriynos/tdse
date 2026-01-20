from tdse import *
from multiprocessing import Pool
import re

boundary_condition = "period"
execute_code("./test3.py", globals())

eigen_waves = load_complex_arrays_from_hdf5("eigen_waves.h5")
eigen_waves_free = load_complex_arrays_from_hdf5("eigen_waves_free.h5")
eigen_waves_all = eigen_waves + eigen_waves_free[::1]

en_list = []
p_list = []

fig, ax = plt.subplots(2, 1)
for i in range(0, len(eigen_waves_all)):
    en = py_get_energy_1d(rt, eigen_waves_all[i])
    # psi_pos, psi_neg, k, psi_k = separate_momentum_components(eigen_waves[i], xgrid)
    # p1 = py_get_kinetic_momentum_1d(rt, psi_pos)
    en_list.append(en)
    # p_list.append(p1)
ax[0].plot(np.real(eigen_waves_all[-1]))
ax[1].plot(en_list)

# plt.show()

dipole_transitions_matrix = np.zeros((len(eigen_waves_all), len(eigen_waves_all)), dtype=complex)
for i in range(0, len(eigen_waves_all)):
    for j in range(0, len(eigen_waves_all)):
        dipole_transitions_matrix[i, j] = py_get_dipole_transition_1d(rt, eigen_waves_all[i], eigen_waves_all[j])

fig, ax = plt.subplots()
ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.25), cmap='hot')
ax.set_title("Dipole Transition Matrix Magnitude")
ax.set_xlabel("Eigenstates")
ax.set_ylabel("Eigenstates")
ax.set_xticks(range(0, len(eigen_waves_all), 10))
ax.set_yticks(range(0, len(eigen_waves_all), 10))
ax.set_xticklabels([f"{en_list[i]:.2f}" for i in range(0, len(en_list), 10)])
ax.set_yticklabels([f"{en_list[i]:.2f}" for i in range(0, len(en_list), 10)])
fig.colorbar(ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.25), cmap='hot'), ax=ax, label='Magnitude')
plt.show()

save_complex_matrix_to_hdf5(dipole_transitions_matrix, "dipole_transitions_matrix.h5")