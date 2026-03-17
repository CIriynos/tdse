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
   
fig, ax = plt.subplots()
ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.5), cmap='hot')
ax.set_title("Dipole Transition Matrix Magnitude")
ax.set_xlabel("Eigenstates")
ax.set_ylabel("Eigenstates")
ax.set_xticks(range(0, len(eigen_waves_all), 20))
ax.set_yticks(range(0, len(eigen_waves_all), 20))
ax.set_xticklabels([f"{en_list[i]:.1f}" for i in range(0, len(en_list), 20)])
ax.set_yticklabels([f"{en_list[i]:.1f}" for i in range(0, len(en_list), 20)])
fig.colorbar(ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.5), cmap='hot'), ax=ax, label='Magnitude')

fig, ax = plt.subplots()
plt.show()