from tdse import *
from multiprocessing import Pool
import re

# thg_data = []

# rate = 2
# Nx = 3600 * rate
# omega0 = 0.057 * 2.0
# delta_x = 0.2 / rate
# delta_t = 0.05 / rate
# nc = 6
# Xi = Nx * delta_x / 2 * 0.8
# a0 = 0.5

# E0 = 0.05 * 5.0
# execute_code("./test.py", globals())

# E0_list = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]) * 0.05 * 0.9

# for E0 in E0_list:
#     execute_code("./test.py", globals())
#     dk = ks[1] - ks[0]
#     thg = hg1[math.floor(3 * omega0 / dk)]
#     thg_data.append(thg)

# plt.figure()
# plt.plot(E0_list ** 2, np.pow(thg_data, 1/3))
# plt.show()


rate = 1
Nx = 5000 * rate
delta_x = 0.2 / rate
delta_t = 0.05 / rate
Xi = Nx * delta_x / 2 * 0.8

omega0 = 0.057
E0 = 0.01
execute_code("./test.py", globals())