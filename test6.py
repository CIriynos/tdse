from tdse import *
from multiprocessing import Pool
import re

wave_list = []
en_list = np.arange(0.01, 2.0, 0.01)

imag_delta_t = 0.1
itp_steps = 200
# boundary_condition = "reflect"
boundary_condition = "period"
bound_flag = 0

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
# ax.set(xlim=(-0.7, 0.0))
ax.set(xlim=(0.0, 3.0))
plt.show()

save_complex_arrays_to_hdf5(wave_list, filename="eigen_waves_free.h5")  # period
# save_complex_arrays_to_hdf5(wave_list, filename="eigen_waves_free_re.h5") # reflect
# save_complex_arrays_to_hdf5(wave_list, filename="eigen_waves_free_re_zero.h5") # reflect + V(x)=0

# save_complex_arrays_to_hdf5(wave_list, filename="eigen_waves_free_hd.h5") # reflect high density
