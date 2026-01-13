from tdse import *
from multiprocessing import Pool
import re

ek_list = [-0.669, -0.510, -0.275, -0.151, -0.093, -0.063, -0.045, -0.035]
ek_list = np.concatenate((ek_list, np.linspace(ek_list[-1], -0.001, 20)))
wave_list = []
en_list = []
last_en = -100.0
for ek in ek_list:
    imag_delta_t = 2 / -ek
    itp_steps = 100
    execute_code("./test3.py", globals())
    if np.abs(last_en - en) > 1e-4 and en < 0:
        wave_list.append(wave.copy())
        en_list.append(en)
    last_en = en

# plot:
print(en_list)
print(len(en_list))
fig, ax = plt.subplots()
ax.eventplot(en_list)
ax.set(xlim=(-0.7, 0.0))
plt.show()

save_complex_arrays_to_hdf5(wave_list, filename="eigen_waves.h5")