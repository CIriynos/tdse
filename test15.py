from tdse import *
from multiprocessing import Pool
import re

second_order_polarization_data = []

E0 = 0.03
execute_code("./test13.py", globals())
sp1 = np.array(pos_expect_spectrum).copy()

E0 = 0
execute_code("./test13.py", globals())
sp2 = np.array(pos_expect_spectrum).copy()

sp = sp1 - sp2

plt.figure()
plt.plot(ks / omega0, np.pow(np.abs(sp1), 2), label="origin spectrum")
plt.plot(ks / omega0, np.pow(np.abs(sp), 2), label="Net spectrum")
plt.xticks(range(1, 30, 2))
plt.grid(True, alpha=0.2)
plt.yscale('log')
plt.ylim(1e-18, 1e2)
plt.legend()
plt.show()