# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tdse import *
import pickle

a = display_ang_freq_SI(0.05)

b = display_electric_field_SI(1.7e-5)

file_name, _ = os.path.splitext(os.path.basename(__file__))
print(file_name + "_" + "storage")
print(globals())

with open('variables.pkl', 'wb') as f:
    pickle.dump(globals(), f)