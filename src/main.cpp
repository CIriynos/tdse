#include <iostream>
#include "tdse.hpp"



int main()
{
    int Nx = 3600;
    double delta_x = 0.2;
    double delta_t = 0.05;
    double imag_delta_t = 0.1;
    double Xi = 200;
    double Lx = Nx * delta_x;

    Grid1D grid(Nx, delta_x, -Lx / 2);
    PhysicalWorld1D world(grid);
    world.potential_data = std::vector<cplx>(Nx, cplx(0.0, 0.0));
    world.absorption_potential_data = std::vector<cplx>(Nx, cplx(0.0, 0.0));
    for(int i = 0; i < Nx; ++i) {
        double x = world.xgrid.get_pos(i);
        world.potential_data[i] = cplx(-std::pow(x*x + 1.0, -0.5));
        world.absorption_potential_data[i] = -100.0 * IM * std::pow((std::abs(x) - Xi) / (Lx / 2 - Xi), 8) * ((std::abs(x) > Xi) ? 1.0 : 0.0);
    }
    // for(int i = 0; i < Nx; ++i) {
    //     double x = world.xgrid.get_pos(i);
    //     world.potential_data[i] = cplx(0.5 * std::pow(x, 2));
    //     world.absorption_potential_data[i] = 0;
    // }

    RuntimeBuffer1D buffer(world, delta_t, imag_delta_t);

    std::vector<cplx> wavefunc = gauss_package_1d(grid, 1.0, 1.0, 0.0);
    imaginary_time_propagation_1d(buffer, wavefunc, 1000);
    
    return 0;
}