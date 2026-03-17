from tdse import *
from multiprocessing import Pool
import re

E0 = 0.03
omega0 = 0.05 * 0.6
sh_phi = 0 * np.pi
a0 = 1.0 * 1
max_display_n = 50
nc = 15

execute_code("./test13.py", globals())