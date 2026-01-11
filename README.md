# How to Use
You can download the wheel file `.whl` and directly import it into your python code.
```shell
pip install tdse-1.0.0-py3-none-any.whl
```
The usage method is no different from other Python libraries
```py
import tdse
```
See Section *Getting Start* for basic usage methods.

# How to Build
This project is a hybrid program based on C++ and Python. It takes into account both computational efficiency and usability. Although you can download the wheel file `.whl` and import it directly, it is recommended to build this project on your own platform, and it is very simple.

The recommended platform is Linux.

## 1. Build C++ Library
Linux: GCC (15.2.0)

Windows: Check [MinGW](https://www.mingw-w64.org/) (A native Windows port of the GNU GCC)

Build from Linux: 
```shell
g++ -O3 -fPIC -shared -static .\src\interface.cpp -o .\tdse\interface.so
```
Build from Windows:

```shell
g++ -O3 -fPIC -shared -static .\src\interface.cpp -o .\tdse\interface.dll
```
The output files (`.dll` in Windows or `.so` in Linux) are collected in `.\tdse`.

## 2. Build Python Package (.whl)

> Note: It is recommended to use virtual python environment (`venv`) for building and installing the package.

Requirement: Python (>3.8)

```shell
pip install build
python -m build --wheel
pip install .\dist\tdse-1.0.0-py3-none-any.whl
```

## 3. Test
It is simple to import the TDSE package in any other python script in different directory
```python
import tdse
```
> [!WARNING] FileNotFoundError: Could not find module '...\interface.dll' ...
> We recommand you build the library with option `-static`, as shown in the instructions before. If not, you may encounter such error. This is because python (>3.8) sometimes fail to find the path for "libstdc++-6.so(.lib)", especially for Windows MinGW environment. System environment variables in `PATH` will not be loaded by python. In other word, you need to manually inform the python interpreter of the locations of those external DLLs (https://docs.python.org/3/library/os.html#os.add_dll_directory). By statically linking these libraries or dependences into your application, you can circumvent this issue.

# Getting Start
Here are the basic concepts:
1. **Physical World**: Define the fundamental space for physical interaction, including space size/dimension, boundary conditions, space grid points, etc. It also defines the potiental function $V(r)$ caused by the Coulomb effect from the atomic nucleus. It is in the level of actual quantum physics.

2. **Runtime Buffer**: According to the given **physical world**, it contains the fundamental elements for algorithm execution, including pre-allocated memory, preset matrices and buffer for intermediate results, etc. It is responsible for the algorithm level.

Here is a template for running 1-Dimension TDSE. All quantities are in atomic unit.
```python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tdse import *

# Define the x-space grid
Nx = 3600               # The number of grid points
delta_x = 0.2           # The grid spacing
Lx = Nx * delta_x       # The length of the x-space grid
# Define the shift of the coordinate of the first grid point (Defualt is zero)
shift_x = -(Nx * delta_x) / 2

# Parameters for evolution
delta_t = 0.05          # The time step
imag_delta_t = 0.1      # The time step for imaginary-time evolution (used for obtaining the ground state)

# T-SURFF Parameters
Xi = 200                # The boundary distance for detecting the free electrons

# Define the laser with an extra DC bias
laser1 = cos2_laser_pulse(delta_t=delta_t, E0=0.053, omega0=0.057, nc=6)
dc = dc_bias(delta_t, Edc=0.0, last_time=2 * (laser1.t_max - laser1.t_min))
laser = combine_light_field(laser1, dc)
# laser.display()       # visualize the optical field

# Define the potential function with absorption boundary
Vx = lambda x: -1.0 / np.sqrt(x * x + 1.0)
Vx_absorb = lambda x: -100j * pow((np.abs(x) - Xi) / (Lx / 2 - Xi), 8) * (np.abs(x) > Xi)

# Create physical world (1d) and runtime buffer
world, xs = create_physical_world_1d(Nx, delta_x, shift_x, Vx, Vx_absorb)
buffer = create_runtime_buffer_1d(world, delta_t, imag_delta_t)

# Calculate the ground state for the given physical world
wave = get_ground_state_1d(world, buffer, 500)

# Execute TDSE propagation (fd: finite different) with auxiliary tsurf results
tsurf_res = tdse_fd1d_tsurf(world, buffer, wave, light_field=laser, Xi=Xi)

# Execute T-SURFF (1d)
ks, Pk = tsurf_1d(tsurf_res, light_field=laser, k_min=-3.0, k_max=3.0, Xi=Xi)

# plot the results (t-surff)
plt.plot(np.sign(ks) * pow(ks, 2) / 2, Pk, lw=0.5)
plt.yscale('log')
plt.ylim(1e-15, 1e-4)
plt.yticks([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
plt.show()
```