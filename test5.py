from tdse import *
from multiprocessing import Pool
import re

execute_code("./test3.py", globals())
eigen_waves = load_complex_arrays_from_hdf5("eigen_waves.h5")
en_list = []

fig, ax = plt.subplots(2, 1)
for i in range(0, len(eigen_waves)):
    ax[0].plot(np.real(eigen_waves[i]))
    en = py_get_energy_1d(rt, eigen_waves[i])
    en_list.append(en)
ax[1].plot((-np.log(-np.array(en_list))) ** 2)

plt.show()
