from tdse import *
from multiprocessing import Pool
import re

wave_list = []
en_list = []
last_en = -100.0
imag_delta_t = 0.1
itp_steps = 200
boundary_condition = "period"
en_list = np.arange(0.01, 3.0, 0.01)

for ekk in en_list:
    imag_delta_t = 2 / -ekk
    itp_steps = 100
    execute_code("./test3.py", globals())
    wave_list.append(wave)

# plot:
print(en_list)
print(len(en_list))
fig, ax = plt.subplots()
ax.eventplot(en_list)
ax.set(xlim=(-0.7, 0.0))
plt.show()

save_complex_arrays_to_hdf5(wave_list, filename="eigen_waves_free.h5")