/*
 * IDEAL.h
 *
 * Copyright 2016 Jerome Lux <jlux@eduroam2058.univ-lr.fr>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 */

#ifndef IDEAL_H
#define IDEAL_H

// Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

// std headers
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>

#include "NSBase.h"

/**
 * @brief IDEAL solver driver (double pressure update loop).
 *
 * Outer loop:
 *  1. assemble the momentum matrices;
 *  2. N1 inner iterations (pressure update): compute the
 *     pseudo-velocities u_hat from the previous velocity, solve the
 *     full pressure system (with under-relaxation) and correct the
 *     velocity with u = u_hat - dp/a;
 *  3. recompute the velocity field from the momentum equation
 *     (with update clamping);
 *  4. N2 inner iterations (final update): compute the
 *     pseudo-velocities from the current velocity, solve the
 *     pressure system (without under-relaxation) and correct the
 *     velocity with u = u_hat - dp/a;
 *  5. adaptive relaxation control (alpha only, beta kept constant).
 *
 * Convergence is based on the relative mass residual (RMR) compared
 * to simpletol.
 *
 * The class holds the working matrices and fields (uA/vA/wA/pA,
 * u/v/w, u_hat/v_hat/w_hat, u_temp/v_temp/w_temp, pvector, pc);
 * the numerical kernels live in NavierStokesBase.
 */
class IDEALSOLVER
{
public:

    // input parameters
    int N1, N2, save_step, maxit;
    double simpletol;

    // momentum matrices (x, y, z components)
    Eigen::SparseMatrix<double, Eigen::RowMajor> uA;
    Eigen::SparseMatrix<double, Eigen::RowMajor> vA;
    Eigen::SparseMatrix<double, Eigen::RowMajor> wA;

    // pressure matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> pA;

    // velocity (current, pseudo and temporary), pressure fields
    Eigen::VectorXd u, u_hat, u_temp;
    Eigen::VectorXd v, v_hat, v_temp;
    Eigen::VectorXd w, w_hat, w_temp;
    Eigen::VectorXd pvector;
    Eigen::VectorXd pc;

    // right-hand side
    Eigen::VectorXd b;

    time_t start, end;

    // member functions
    /**
     * @brief Constructor.
     *
     * @param simpletol convergence threshold on the relative RMR.
     * @param maxit     maximum number of outer iterations.
     * @param N1        number of inner pressure iterations (step 2).
     * @param N2        number of inner final iterations (step 4).
     */
    IDEALSOLVER(double simpletol, int maxit, int N1, int N2);

    /**
     * @brief Run the IDEAL algorithm.
     *
     * Initializes the fields, runs the outer loop described in the
     * class documentation and writes the final fields to VTI files.
     *
     * @param NSBase base class instance (geometry, solvers, kernels).
     */
    void Solve(NavierStokesBase & NSBase);

    // NOTE: the following two methods were declared here but are never defined
    // nor called (the real ones are NavierStokesBase::CheckConvergence and
    // NavierStokesBase::Rotation), so they have been commented out:
    //   bool CheckConvergence(double & res);
    //   void Rotation(int direction);
};

#endif
