#ifndef __INTERFACE_H__
#define __INTERFACE_H__

#include "tdse.hpp"

#ifdef _MSC_VER
    #define DLL_EXPORT __declspec( dllexport ) 
#else
    #define DLL_EXPORT
#endif

extern "C"
{
    DLL_EXPORT int add(int a, int b);

    DLL_EXPORT void * create_physical_world_1d(int Nx, double delta_x, double shift_x,
        double * potential, double * absorb_potential_real, double * absorb_potential_imag);

    DLL_EXPORT int test(void * world_p);

    DLL_EXPORT void * create_runtime_buffer_1d(void * world_p, double delta_t, double delta_t_imag);

    DLL_EXPORT void * get_ground_state_1d(void * buffer_p, int time_steps);

    DLL_EXPORT double get_energy_1d(void * buffer_p, void * wavefunc);

    DLL_EXPORT double get_pos_expect_1d(void * wd_p, void * wavefunc);

    DLL_EXPORT double get_accel_expect_1d(void * wd_p, void * wavefunc);

    DLL_EXPORT double * get_norm_1d(void * wd_p, void * wavefunc);

    DLL_EXPORT void tdse_laser_fd1d_onestep(void * buffer_p, void * wavefunc, double At);

    DLL_EXPORT double * get_wave_value_1d(void * wd_p, void * wavefunc, double x_pos);

    DLL_EXPORT double * get_wave_1diff_value_1d(void * wd_p, void * wavefunc, double x_pos);
}


#endif //__INTERFACE_H__