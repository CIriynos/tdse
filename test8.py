from tdse import *
from multiprocessing import Pool
import re
import numpy as np
import matplotlib.pyplot as plt

ek = 2.0
# ek = -0.001
imag_delta_t = 2 / -ek
itp_steps = 200
boundary_condition = "reflect"
execute_code("./test3.py", globals())

print(np.abs(np.vdot(wave, xgrid * wave)))
# wave *= np.exp(-xgrid ** 2 / 500 ** 2)
print(np.abs(np.vdot(wave, xgrid * wave)))
print(np.abs(np.trapezoid(np.conj(wave) * xgrid * wave)))

plt.plot(xgrid, np.real(wave))

# plt.plot(xgrid, np.real(np.conj(wave) * xgrid * wave))
# plt.plot(xgrid, -np.flip(np.real(np.conj(wave) * xgrid * wave)))
plt.show()