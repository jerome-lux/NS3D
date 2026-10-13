/*
 * SIMPLEC.h
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

#ifndef SIMPLEC_H
#define SIMPLEC_H

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
 * @brief SIMPLEC solver driver.
 *
 * Outer loop:
 *  1. assemble the momentum matrices and compute the velocity field
 *     (with update clamping);
 *  2. assemble and solve the SIMPLEC pressure correction system
 *     (row sums of the momentum matrices);
 *  3. apply the velocity correction (SIMPLEC formulation) and the
 *     pressure update p += beta * pc;
 *  4. adaptive relaxation control.
 *
 * Convergence is based on the relative mass residual (RMR) compared
 * to simpletol.
 *
 * The class holds the working matrices and fields (uA/vA/wA/pA,
 * u/v/w, pvector, pc); the numerical kernels live in NavierStokesBase.
 */
class SIMPLECSOLVER
{
public:

    // input parameters
    int save_step, maxit;
    double simpletol;

    // momentum matrices (x, y, z components)
    Eigen::SparseMatrix<double, Eigen::RowMajor> uA;
    Eigen::SparseMatrix<double, Eigen::RowMajor> vA;
    Eigen::SparseMatrix<double, Eigen::RowMajor> wA;

    // pressure correction matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> pA;

    // velocity, pressure and pressure correction fields
    Eigen::VectorXd u, u_temp;
    Eigen::VectorXd v, v_temp;
    Eigen::VectorXd w, w_temp;
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
     */
    SIMPLECSOLVER(double simpletol, int maxit);

    /**
     * @brief Run the SIMPLEC algorithm.
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
