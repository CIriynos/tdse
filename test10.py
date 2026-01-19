from tdse import *
from multiprocessing import Pool
import re
    
wave_list_1 = []
wave_list_2 = []
en_list = np.arange(0.01, 2.0, 0.01)

# b0 = 0
imag_delta_t = -100.0
itp_steps = 200
Nx = 5000 * 2
boundary_condition = "period"
execute_code("./test3.py", globals())

for ekk in en_list:
    imag_delta_t = 2 / -ekk
    itp_steps = 100
    execute_code("./test3.py", globals())
    wave_list_1.append(psi_odd)
    wave_list_2.append(psi_even)

wave_list = wave_list_1 + wave_list_2

# plot:
print(en_list)
print(len(en_list))
fig, ax = plt.subplots()
ax.eventplot(en_list)
# ax.set(xlim=(-0.7, 0.0))
ax.set(xlim=(0.0, 3.0))
# plt.show()

save_complex_arrays_to_hdf5(wave_list, filename="test10.h5")


##############################
eigen_waves_all = load_complex_arrays_from_hdf5("test10.h5")

en_list = []
fig, ax = plt.subplots(2, 1)
for i in range(0, len(eigen_waves_all)):
    en = py_get_energy_1d(rt, eigen_waves_all[i])
    en_list.append(en)
ax[0].plot(np.real(eigen_waves_all[-1]))
ax[1].plot(en_list)

dipole_transitions_matrix = np.zeros((len(eigen_waves_all), len(eigen_waves_all)), dtype=complex)
for i in range(0, len(eigen_waves_all)):
    for j in range(0, len(eigen_waves_all)):
        dipole_transitions_matrix[i, j] = py_get_dipole_transition_1d(rt, eigen_waves_all[i], eigen_waves_all[j])

fig, ax = plt.subplots()

ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.25), cmap='hot')
ax.set_title("Dipole Transition Matrix Magnitude")
ax.set_xlabel("Eigenstates")
ax.set_ylabel("Eigenstates")
space = 20
ax.set_xticks(range(0, len(eigen_waves_all), space))
ax.set_yticks(range(0, len(eigen_waves_all), space))
ax.set_xticklabels([f"{en_list[i]:.2f}" for i in range(0, len(en_list), space)])
ax.set_yticklabels([f"{en_list[i]:.2f}" for i in range(0, len(en_list), space)])
fig.colorbar(ax.imshow(np.pow(np.abs(dipole_transitions_matrix), 0.25), cmap='hot'), ax=ax, label='Magnitude')

print(dipole_transitions_matrix[0, 1])
print(dipole_transitions_matrix[-1, -2])

plt.show()  