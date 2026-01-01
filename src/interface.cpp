#include "interface.h"

extern "C"
{
    int add(int a, int b)
    {
        return a + b;
    }

    DLL_EXPORT
    void * create_physical_world_1d(int Nx, double delta_x, double shift_x,
        double potential[], double absorb_potential_real[], double absorb_potential_imag[])
    {
        Grid1D grid(Nx, delta_x, shift_x);
        PhysicalWorld1D * world_p = new PhysicalWorld1D(grid);
        for(int i = 0; i < Nx; ++i) {
            double x = grid.get_pos(i);
            world_p->potential_data[i] = potential[i];
            world_p->absorption_potential_data[i] = absorb_potential_real[i] + IM * absorb_potential_imag[i];
        }
        return (void*) (world_p);
    }

    DLL_EXPORT 
    int test(void * world) {
        return (*(PhysicalWorld1D *)world).xgrid.N;
    }

    DLL_EXPORT 
    void * create_runtime_buffer_1d(void * wd_p, double delta_t, double imag_delta_t)
    {
        PhysicalWorld1D * world_p = (PhysicalWorld1D *) wd_p;
        RuntimeBuffer1D * buffer_p = new RuntimeBuffer1D(*world_p, delta_t, imag_delta_t);
        return (void*) (buffer_p);
    }

    DLL_EXPORT 
    void * get_ground_state_1d(void * wd_p, void * buffer_p, int time_steps)
    {
        PhysicalWorld1D * world_p = (PhysicalWorld1D *) wd_p;
        std::vector<cplx> wavefunc = gauss_package_1d(world_p->xgrid, 1.0, 1.0, 0.0);
        imaginary_time_propagation_1d(*(RuntimeBuffer1D*)buffer_p, wavefunc, time_steps);
        std::vector<cplx> * result = new std::vector<cplx>(wavefunc);
        return (void *) result;
    }

    DLL_EXPORT
    double get_energy_1d(void * buffer_p, void * wavefunc)
    {
        cplx energy = get_energy_1d(*(RuntimeBuffer1D*)buffer_p, *(std::vector<cplx>*) wavefunc);
        return (double)energy.real();
    }

    DLL_EXPORT
    void tdse_laser_fd1d_onestep(void * buffer_p, void * wavefunc, double At)
    {
        tdse_laser_fd1d_onestep(*(RuntimeBuffer1D*)buffer_p, *(std::vector<cplx>*) wavefunc, (*(RuntimeBuffer1D*)buffer_p).delta_t, At);
    }

    double * convert_cplx_to_array2(const cplx& num)
    {
        double * num_c_complex = new double[2];
        num_c_complex[0] = (double)num.real();
        num_c_complex[1] = (double)num.imag();
        return num_c_complex;
    }

    DLL_EXPORT
    double * get_norm_1d(void * buffer_p, void * wavefunc)
    {
        return convert_cplx_to_array2(get_norm_1d(*(std::vector<cplx>*) wavefunc));
    }

    DLL_EXPORT 
    double * get_wave_value_1d(void * wd_p, void * wavefunc, double x_pos)
    {
        int x_id = ((PhysicalWorld1D*)wd_p)->xgrid.index(x_pos);
        return convert_cplx_to_array2((*(std::vector<cplx>*)wavefunc)[x_id]);
    }

    DLL_EXPORT 
    double * get_wave_1diff_value_1d(void * wd_p, void * wavefunc, double x_pos)
    {
        int x_id = ((PhysicalWorld1D*)wd_p)->xgrid.index(x_pos);
        cplx num1 = (*(std::vector<cplx>*)wavefunc)[x_id - 1];
        cplx num2 = (*(std::vector<cplx>*)wavefunc)[x_id + 1];
        return convert_cplx_to_array2(-(num1 - num2) / (2.0 * ((PhysicalWorld1D*)wd_p)->xgrid.get_delta()));
    }
}