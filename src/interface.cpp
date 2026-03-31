#include "interface.hpp"

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
    void * get_ground_state_1d(void * buffer_p, int time_steps)
    {
        std::vector<cplx> wavefunc = gauss_package_1d(((RuntimeBuffer1D*)buffer_p)->world.xgrid, 1.0, 1.0, 0.0);
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
    double get_pos_expect_1d(void * world_p, void * wavefunc)
    {
        cplx pos_expect = get_pos_expect_1d(*(PhysicalWorld1D*)world_p, *(std::vector<cplx>*) wavefunc);
        return (double)pos_expect.real();
    }

    DLL_EXPORT
    double get_pos_expect_1d_masked(void * world_p, void * wavefunc, double mask_sigma)
    {
        cplx pos_expect = get_pos_expect_1d_masked(*(PhysicalWorld1D*)world_p, *(std::vector<cplx>*) wavefunc, mask_sigma);
        return (double)pos_expect.real();
    }

    DLL_EXPORT 
    double get_pos_expect_cross_1d(void * world_p, void * wavefunc_free, void * wavefunc_bound)
    {
        cplx pos_expect = get_pos_expect_cross_1d(*(PhysicalWorld1D*)world_p, *(std::vector<cplx>*) wavefunc_free, *(std::vector<cplx>*) wavefunc_bound);
        return (double)pos_expect.real();
    }

    DLL_EXPORT 
    double get_accel_expect_1d(void * world_p, void * wavefunc)
    {
        cplx accel_expect = get_accel_expect_1d(*(PhysicalWorld1D*)world_p, *(std::vector<cplx>*) wavefunc);
        return (double)accel_expect.real();
    }

    double * convert_cplx_to_array2(const cplx& num)
    {
        double * num_c_complex = new double[2];
        num_c_complex[0] = (double)num.real();
        num_c_complex[1] = (double)num.imag();
        return num_c_complex;
    }

    DLL_EXPORT 
    double * project_out_bound_state_1d(void * world_p, void * wavefunc, void * bound_state)
    {
        cplx res = project_out_bound_state_1d(*(PhysicalWorld1D*)world_p, *(std::vector<cplx>*) wavefunc, *(std::vector<cplx>*) bound_state);
        return convert_cplx_to_array2(res);
    }

    DLL_EXPORT
    void tdse_laser_fd1d_onestep(void * buffer_p, void * wavefunc, double At)
    {
        tdse_laser_fd1d_onestep(*(RuntimeBuffer1D*)buffer_p, *(std::vector<cplx>*) wavefunc, (*(RuntimeBuffer1D*)buffer_p).delta_t, At);
    }

    DLL_EXPORT
    double * get_norm_1d(void * wd_p, void * wavefunc)
    {
        return convert_cplx_to_array2(get_norm_1d(*(PhysicalWorld1D*) wd_p, *(std::vector<cplx>*) wavefunc));
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

    DLL_EXPORT
    void get_wave_value_list_1d(void * wd_p, void * wavefunc, double * wave_real, double * wave_imag)
    {
        int Nx = ((PhysicalWorld1D*)wd_p)->xgrid.N;
        std::vector<cplx>& wave = *(std::vector<cplx>*) wavefunc;
        for(int i = 0; i < Nx; i++) {
            wave_real[i] = wave[i].real();
            wave_imag[i] = wave[i].imag();
        }
    }

    DLL_EXPORT
    void * get_wave_copy(void * wd_p, void * wavefunc)
    {
        std::vector<cplx>& wave = *(std::vector<cplx>*) wavefunc;
        std::vector<cplx> * wave_copy = new std::vector<cplx>(wave);
        return (void *) wave_copy;
    }

    DLL_EXPORT
    void superimpose_wave(void * wd_p, void * wavefunc, void * added_wave, double coeff_real, double coeff_imag)
    {
        std::vector<cplx>& wave = *(std::vector<cplx>*) wavefunc;
        std::vector<cplx>& added_wave_vec = *(std::vector<cplx>*) added_wave;
        cplx coeff(coeff_real, coeff_imag);
        for(int i = 0; i < wave.size(); i++) {
            wave[i] += coeff * added_wave_vec[i];
        }
    }

    DLL_EXPORT
    void * get_empty_wave(void * wd_p)
    {
        int Nx = ((PhysicalWorld1D*)wd_p)->xgrid.N;
        std::vector<cplx> * empty_wave = new std::vector<cplx>(Nx, cplx(0.0, 0.0));
        return (void *) empty_wave;
    }

    DLL_EXPORT
    void transform_to_length_gauge(void * wd_p, void * wavefunc, double At)
    {
        std::vector<cplx>& wave = *(std::vector<cplx>*) wavefunc;
        PhysicalWorld1D& world = *(PhysicalWorld1D*) wd_p;
        for(int i = 0; i < wave.size(); i++) {
            double x = world.xgrid.get_pos(i);
            wave[i] *= exp(IM * At * x);
        }
    }
}