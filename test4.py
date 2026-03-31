from tdse import *
from multiprocessing import Pool
import re

a0 = 2.0
short_range_flag = True
short_range_sigma = 5.0
def query_func(ek):
    global imag_delta_t, itp_steps
    print(f"Calculating for ek = {ek} ...")
    imag_delta_t = 2 / -ek
    itp_steps = 100
    execute_code("./test3.py", globals())
    return en

wave_list = []
en_list = []
last_en = -100.0
imag_delta_t = 0.1
itp_steps = 200

en_list = find_all_points_with_error(-0.7, -0.0007, query_func, delta=0.7e-6)

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

# save_complex_arrays_to_hdf5(wave_list, filename="eigen_waves.h5")