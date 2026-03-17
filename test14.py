from tdse import *
from multiprocessing import Pool
import re

second_order_polarization_data = []

E0 = 0.05
execute_code("./test13.py", globals())

E0_list = np.linspace(0.1, 1.2, 12) * 0.05

for E0 in E0_list:
    execute_code("./test13.py", globals())
    dk = ks[1] - ks[0]
    tmp = pos_expect_spectrum[math.floor(2 * omega0 / dk)]
    second_order_polarization_data.append(tmp)

save_complex_arrays_to_hdf5([np.array(second_order_polarization_data)], filename="test14.h5")

###
data = load_complex_arrays_from_hdf5("test14.h5")

plt.figure()
plt.plot(E0_list ** 2, np.pow(np.abs(data[0]), 2/2))
plt.scatter(E0_list ** 2, np.pow(np.abs(data[0]), 2/2))
plt.show()