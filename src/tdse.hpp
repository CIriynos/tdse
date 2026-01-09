#ifndef __TDSE_HPP__
#define __TDSE_HPP__

#include <vector>
#include <complex>
#include <cassert>
#include <iostream>
#include "util.hpp"

// /**
//  * @brief Configuration parameters for the TDSE solver.
//  */
// class Configuration1D {
// public:
//     int Nx;             // Number of grid points in x-direction
//     double delta_x;     // Grid spacing in x-direction    
//     double delta_t;     // Time step size
//     double imag_delta_t;   // time step size for imaginary time evolution
// };

/**
 *  @brief 1-dimension uniform spacing grid for a certain physical coordinate. 
 *      Two concepts: coordinate & nodes. 
 *      Nodes are sampled points consisting of the whole numerical grid.
 *      Coordinate represents the actual (or physical) space / time domain.
 */
class Grid1D {
public:
    int N;             // Number of grid points
    double delta;     // Grid spacing
    double shift;     // Spatial shift

    // Grid1D(int N, double left_bound, double right_bound)
    //     : N(N), delta((right_bound - left_bound) / N), shift(left_bound)
    // {
    //     assert(right_bound > left_bound);
    // }
    
    Grid1D(int N, double delta, double shift)
        : N(N), delta(delta), shift(shift)
    {
    }

    double get_delta() const { return delta; }
    double get_number() const { return N; }
    double get_length() const { return N * delta; }
    double get_first_pos() const { return shift; }
    double get_last_pos() const { return shift + get_length() - delta;}

    std::vector<double> get_grid_data() const {
        std::vector<double> grid_data;
        grid_data.reserve(N);
        for(int i = 0; i < N; ++i) {
            grid_data.push_back(shift + i * delta);
        }
        return grid_data;
    }

    /**
     * @brief Get the index corresponding to a given coordinate.
     * @param pos coordinate position
     * @return Index in the grid
     */
    int index(double pos) const {
        int id = static_cast<int>((pos - shift) / delta);
        assert(id >= 0 && id < N);  // Index out of bounds in Configuration1D::index()
        return id;
    }

    /**
     * @brief Get the coordinate corresponding to a given index.
     * @param index Index in the grid
     * @return coordinate
     */
    double get_pos(int index) const {
        return shift + index * delta;
    }
};

/**
 * @brief Physical world parameters
 */
struct PhysicalWorld1D {
    Grid1D xgrid;
    std::vector<cplx> potential_data;
    std::vector<cplx> absorption_potential_data;

    PhysicalWorld1D(const Grid1D& xgrid)
        : xgrid(xgrid), potential_data(xgrid.get_number()),
        absorption_potential_data(xgrid.get_number())
    {
    }
    

};


/**
 * @brief Runtime buffer for TDSE solver, including pre
 */
class RuntimeBuffer1D {
public:
    // basic parameters
    double Nx;
    double delta_x;
    double delta_t;
    double imag_delta_t;

    PhysicalWorld1D world;

    // basic matrices in TDSE solver
    PentaDiagonalMatrix D2;             // Second derivative matrix
    PentaDiagonalMatrix D1;             // First derivative matrix
    PentaDiagonalMatrix I;              // Identity matrix
    PentaDiagonalMatrix A_pos, A_neg;   // Matrices for Crank-Nicolson scheme
    PentaDiagonalMatrix H, H_absorb;    // Hamiltonian matrices

    // For imaginary time propagation (ITP)
    PentaDiagonalMatrix A_pos_itp, A_neg_itp;

    // runtime temporary matrices (pre allocated for efficiency)
    std::vector<PentaDiagonalMatrix> temp_matrices;
    std::vector<std::vector<cplx> > temp_vectors;

    /**
     * @brief Constructor to initialize runtime buffer based on physical world and configuration.
     * @param config Configuration of TDSE.
     * @param world Physical world parameters
     */
    RuntimeBuffer1D(const PhysicalWorld1D& world, double delta_t, double imag_delta_t)
        : Nx(world.xgrid.get_number()), 
          delta_x(world.xgrid.get_delta()),
          delta_t(delta_t), imag_delta_t(imag_delta_t),
          world(world),
          D2(world.xgrid.get_number(), -1, 16, -30, 16, -1),
          D1(world.xgrid.get_number(), 1, -8, 0, 8, -1),
          I(world.xgrid.get_number(), 0, 0, 1, 0, 0),
          A_pos(world.xgrid.get_number()), A_neg(world.xgrid.get_number()),
          H(world.xgrid.get_number()), H_absorb(world.xgrid.get_number()),
          A_pos_itp(world.xgrid.get_number()), A_neg_itp(world.xgrid.get_number())
    {
        for (int i = 0; i < num_temp_matrices; ++i) {
            temp_matrices.emplace_back(Nx);
        }
        for (int i = 0; i < num_temp_vectors; ++i) {
            temp_vectors.emplace_back(std::vector<cplx>(Nx, cplx(0.0, 0.0)));
        }

        PentaDiagonalMatrix V(Nx, world.potential_data);
        PentaDiagonalMatrix V_only_absorb(Nx, world.absorption_potential_data);
        PentaDiagonalMatrix V_absorb = V + V_only_absorb;

        D2 *= (1.0 / (12.0 * delta_x * delta_x));
        D1 *= (1.0 / (12.0 * delta_x));

        A_pos = I + (0.25 * IM * delta_t) * D2 + (-0.5 * IM * delta_t) * V_absorb;
        A_neg = I + (-0.25 * IM * delta_t) * D2 + (0.5 * IM * delta_t) * V_absorb;
        H = -0.5 * D2 + V;
        H_absorb = -0.5 * D2 + V_absorb;

        A_pos_itp = I + (0.25 * imag_delta_t) * D2 + (-0.5 * imag_delta_t) * V;
        A_neg_itp = I + (-0.25 * imag_delta_t) * D2 + (0.5 * imag_delta_t) * V;
    }

private:
    static const int num_temp_matrices = 5;         // number of temporary matrices
    static const int num_temp_vectors = 5;           // number of temporary vectors
};

/**
 * @brief Generate a Gaussian wave packet in 1D.
 * @param config Configuration parameters
 * @param omega Width of the Gaussian
 * @param k0 Central wave number
 * @param x0 Central position
 * @return Vector containing the Gaussian wave packet
 */
inline std::vector<cplx> gauss_package_1d(const Grid1D& grid, double omega, double k0, double x0) {
    std::vector<cplx> result(grid.get_number(), cplx(0.0, 0.0));
    for(int i = 0; i < grid.get_number(); ++i) {
        double x = grid.get_pos(i);
        result[i] = (1.0 / std::pow(2 * PI, 0.25)) * std::exp(IM * k0 * x) * std::exp(-std::pow((x - x0) / (2 * omega), 2));
    }
    return result;
}

/**
 * @brief Main loop for the TDSE solver using finite differences in 1D.
 * @param config Configuration parameters
 * @param buffer Runtime buffer for pre-allocated matrices and vectors
 * @param wavefunc Current wavefunction
 * @param num_time_steps Number of time steps to simulate
 */
inline void tdse_fd1d_mainloop(RuntimeBuffer1D& buffer, std::vector<cplx>& wavefunc, int num_time_steps) {
    std::vector<cplx>& temp_vec1 = buffer.temp_vectors[0];
    std::vector<cplx>& temp_vec2 = buffer.temp_vectors[1];

    for(int step = 0; step < num_time_steps; ++step) {
        // Compute RHS: temp_vec1 = A_neg * wavefunc
        temp_vec1 = buffer.A_pos * wavefunc;
        // Solve LHS: A_pos * wavefunc_new = temp_vec1
        solve_linear_system(buffer.temp_matrices[0], buffer.temp_vectors[2], buffer.A_neg, temp_vec1, wavefunc);
    }
}

/**
 * @brief Compute the norm of a wavefunction in 1D.
 * @param config Configuration parameters
 * @param wavefunc Current wavefunction
 * @return Norm of the wavefunction
 */
inline cplx get_norm_1d(const PhysicalWorld1D& world, const std::vector<cplx>& wavefunc) {
    cplx norm = cplx(0.0, 0.0);
    for (const auto& val : wavefunc) {
        norm += std::conj(val) * val;
    }
    return norm * world.xgrid.delta;
}

/**
 * @brief Compute the energy of a wavefunction in 1D.
 * @param world PhysicalWorld parameters
 * @param buffer Runtime buffer for pre-allocated matrices and vectors
 * @param wavefunc Current wavefunction
 * @return Energy of the wavefunction
 */
inline cplx get_energy_1d(const RuntimeBuffer1D& buffer, const std::vector<cplx>& wavefunc) {
    std::vector<cplx> H_psi = buffer.H * wavefunc;
    cplx energy = cplx(0.0, 0.0);
    for(int i = 0; i < buffer.Nx; ++i) {
        energy += std::conj(wavefunc[i]) * H_psi[i];
    }
    return energy * buffer.delta_x;
}

inline cplx get_pos_expect_1d(const PhysicalWorld1D& world, const std::vector<cplx>& wavefunc) {
    cplx pos_expect = cplx(0.0, 0.0);
    for(int i = 0; i < world.xgrid.N; ++i) {
        pos_expect += std::conj(wavefunc[i]) * wavefunc[i] * world.xgrid.get_pos(i);
    }
    return pos_expect * world.xgrid.delta;
}

inline cplx get_accel_expect_1d(const PhysicalWorld1D& world, const std::vector<cplx>& wavefunc) {
    cplx accel_expect = cplx(0.0, 0.0);
    auto dU = get_diff_data_2o(world.potential_data, world.xgrid.delta);
    for(int i = 0; i < world.xgrid.N; ++i) {
        accel_expect += std::conj(wavefunc[i]) * wavefunc[i] * -dU[i];
    }
    return accel_expect * world.xgrid.delta;
}

/**
 * @brief Perform imaginary time propagation for a wavefunction in 1D.
 *        It is a method to find the ground state of a quantum system.
 * @param config Configuration parameters
 * @param buffer Runtime buffer for pre-allocated matrices and vectors
 * @param wavefunc Current wavefunction
 * @param num_time_steps Number of time steps to simulate
 */
inline void imaginary_time_propagation_1d(RuntimeBuffer1D& buffer, std::vector<cplx>& wavefunc, int num_time_steps) {
    std::vector<cplx>& temp_vec1 = buffer.temp_vectors[0];
    std::vector<cplx>& temp_vec2 = buffer.temp_vectors[1];

    for(int step = 0; step < num_time_steps; ++step) {
        // logging
        if (step % 100 == 0) {
            auto energy = get_energy_1d(buffer, wavefunc);
            auto norm = get_norm_1d(buffer.world, wavefunc);
            std::cout << "ITP Step " << step << ": Energy = " << energy << ", Norm = " << norm << std::endl;
        }

        temp_vec1 = buffer.A_pos_itp * wavefunc;    // Compute RHS: temp_vec1 = A_pos_itp * wavefunc
        solve_linear_system(buffer.temp_matrices[0], temp_vec2, buffer.A_neg_itp, temp_vec1, wavefunc); // Solve LHS: A_neg_itp * wavefunc_new = temp_vec1

        // Normalize wavefunction
        cplx norm = get_norm_1d(buffer.world, wavefunc);
        for(auto& val : wavefunc) { 
            val /= std::sqrt(norm); 
        }
    }
}


inline void tdse_laser_fd1d_onestep(
    RuntimeBuffer1D& buffer, std::vector<cplx>& wavefunc,
    double delta_t,
    double At      // coupled laser field At_data (vector potential)
    ) 
{
    std::vector<cplx>& temp_vec1 = buffer.temp_vectors[0];
    std::vector<cplx>& temp_vec2 = buffer.temp_vectors[1];

    buffer.temp_matrices[0] = buffer.A_pos + (-0.5) * delta_t * At * buffer.D1;
    buffer.temp_matrices[1] = buffer.A_neg + 0.5 * delta_t * At * buffer.D1;
    temp_vec1 = buffer.temp_matrices[0] * wavefunc;
    solve_linear_system(buffer.temp_matrices[2], temp_vec2, buffer.temp_matrices[1], temp_vec1, wavefunc);
}

template<typename T>
inline T hanning_window(T t, T max_t)
{
    return 0.5 * (1 - std::cos(2 * PI * t / max_t));
}

inline cplx tsurf_1d(RuntimeBuffer1D& buffer, 
    double k,
    const Grid1D& t_grid,
    const std::vector<double>& At_data,
    double Xi,
    const std::vector<cplx>& X_pos_vals,
    const std::vector<cplx>& X_neg_vals,
    const std::vector<cplx>& X_pos_dvals,
    const std::vector<cplx>& X_neg_dvals
)
{
    cplx b1k = 0.0;
    cplx b2k = 0.0;
    cplx alpha = 0.0;

    for(int i = 0; i < t_grid.get_number(); i++){
        double t = t_grid.get_pos(i);
        alpha += At_data[i] * buffer.delta_t;
        b1k += hanning_window(t, t_grid.get_length()) * (buffer.delta_t / std::sqrt(2 * PI)) 
            * std::exp(IM * t * k * k / 2.0) * std::exp(-IM * k * (Xi - alpha))
            * ((0.5 * k + At_data[i]) * X_pos_vals[i] - 0.5 * IM * X_pos_dvals[i]);
        b2k += hanning_window(t, t_grid.get_length()) * (-buffer.delta_t / std::sqrt(2 * PI)) 
            * std::exp(IM * t * k * k / 2.0) * std::exp(-IM * k * (-Xi - alpha))
            * ((0.5 * k + At_data[i]) * X_neg_vals[i] - 0.5 * IM * X_neg_dvals[i]);
    }
    return b1k + b2k;
}


// inline void tdse_laser_fd1d_mainloop(
//     const Configuration1D& config, RuntimeBuffer1D& buffer, std::vector<cplx>& wavefunc,
//     A
//     )
// {

// } 


#endif // __TDSE_HPP__
