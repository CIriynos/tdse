from tdse import *
import re

boundary_condition = "reflect"
execute_code("./test3.py", globals())

eigen_waves = load_complex_arrays_from_hdf5("eigen_waves.h5")
eigen_waves_free = load_complex_arrays_from_hdf5("eigen_waves_free_hd.h5")
eigen_waves_all = eigen_waves + eigen_waves_free[:]


en_list = []
fig, ax = plt.subplots(2, 1)
for i in range(0, len(eigen_waves_all)):
    en = py_get_energy_1d(rt, eigen_waves_all[i])
    en_list.append(en)
ax[0].plot(np.real(eigen_waves_all[-1]))
ax[1].plot(en_list)

ionized_pack = gauss_pkg_f(xgrid, 5.0, 0.2, 200.0)
ionized_pack /= np.sqrt(np.vdot(ionized_pack, ionized_pack))
ionized_pack_2 = project_out(rt, ionized_pack, eigen_waves)
coeffs = get_coefficients(rt, ionized_pack, eigen_waves_all)
coeffs_2 = get_coefficients(rt, ionized_pack_2, eigen_waves_all)
ionized_pack_3 = reconstruct_wave(rt, coeffs_2, eigen_waves_all)
coeffs_3 = get_coefficients(rt, ionized_pack_3, eigen_waves_all)
print(np.vdot(coeffs, coeffs))
print(np.vdot(coeffs_2, coeffs_2))
print(np.vdot(coeffs_3, coeffs_3))

fig, ax = plt.subplots(1, 1)
plt.plot(xgrid, np.real(ionized_pack), label='Original Wave Packet')
plt.plot(xgrid, np.real(ionized_pack_2), label='After Projecting Out Bound States')
plt.plot(xgrid, np.real(ionized_pack_3), label='Reconstructed Wave Packet from Coefficients', linestyle='--')

fig, ax = plt.subplots(1, 1)
plt.plot(en_list, np.abs(coeffs))
plt.plot(en_list, np.abs(coeffs_2), '--')
plt.plot(en_list, np.abs(coeffs_3), ':')
plt.title("Projection Coefficients onto Bound States")
plt.xlabel("Eigenstate Index")
plt.ylabel("Coefficient Magnitude")
plt.show()