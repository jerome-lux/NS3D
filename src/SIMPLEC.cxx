/*
 * SIMPLEC.cxx
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

#define NDEBUG
#include <assert.h>

// OpenMP (used by the NSBase kernels)
#include <omp.h>

// disable the Eigen debug checks: the definition must come before
// the Eigen headers are included to have any effect.
#define EIGEN_NO_DEBUG

// Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

// Blitz++ for 3D arrays
#include <blitz/array.h>

// std headers
#include <cstdio>
#include <cmath>
#include <fstream>
#include <string>
#include <ctime>

// project headers
#include "SIMPLEC.h"
#include "NSBase.h"

/**
 * @brief Construct the SIMPLEC solver.
 *
 * @param simpletol convergence threshold on the relative RMR.
 * @param maxit     maximum number of outer iterations.
 */
SIMPLECSOLVER::SIMPLECSOLVER(double simpletol, int maxit) :
                        maxit(maxit), simpletol(simpletol)
{
    start = time(0);
}

/**
 * @brief Run the SIMPLEC algorithm.
 *
 * See the class documentation in SIMPLEC.h for the loop structure.
 * The working fields are initialized, the initial velocity field is
 * computed, then the outer loop runs until convergence (or maxit),
 * saving intermediate fields every save_step iterations, and finally
 * the fields are written to VTI files.
 *
 * @param NSBase base class instance (geometry, solvers, kernels).
 */
void SIMPLECSOLVER::Solve(NavierStokesBase & NSBase)
{
    bool convergence; // 'relax' was declared here but never used in SIMPLEC (only AssemblePressure uses it, which SIMPLEC does not call)
    double Q = 0, v_moy = 0, K = 0;
    double normdiv, reldiv, velocityNorm, maxdiv, MaxDeltaV;
    std::string outfilename;
    int nu = NSBase.nu, nv = NSBase.nv, nw = NSBase.nw, np = NSBase.np;

    int it; // 'N0' (=20) and 'counter' were declared here but never used
    std::string savefile;

    save_step = NSBase.save_step; // was hardcoded to 20; now taken from the input file
    int savecounter = 0;

    printf("Initializing SIMPLEC algorithm\n");
    NSBase.info_file << "Initializing SIMPLEC algorithm" << std::endl;

    // allocate the velocity and pressure fields
    u.resize(nu);
    u = Eigen::VectorXd::Zero(nu);
    v.resize(nv);
    v = Eigen::VectorXd::Zero(nv);
    w.resize(nw);
    w = Eigen::VectorXd::Zero(nw);

    // Eigen::VectorXd temp; // removed: local 'temp' was never used (only temp.resize(0))

    pvector.resize(np);
    pvector = Eigen::VectorXd::Zero(np);
    pc.resize(np);
    pc = Eigen::VectorXd::Zero(np);

    NSBase.Initpvector(NSBase.pfieldfilename, pvector);
    NSBase.InitVelocity(u, v, w);

    // set up the solvers
    NSBase.psolver.setMaxIterations(NSBase.maxitsolver);
    NSBase.psolver.setTolerance(NSBase.psolvertol);
    NSBase.usolver.setTolerance(NSBase.vsolvertol);
    NSBase.vsolver.setTolerance(NSBase.vsolvertol);
    NSBase.wsolver.setTolerance(NSBase.vsolvertol);

    NSBase.info_file << "Max number of SIMPLE iterations: " << maxit << std::endl;
    printf("Stopping criterion for relative residual: %g\n", simpletol);

    NSBase.log_file << "iteration, Q (m³/s), <u_out> (m/s), Max Rel Mass Res., K (m²), gradP (Pa/m), ||Div_u||, Lref*(max DIV)/<u_out>, <||u||>, max DIV, it_u,it_v,it_w,it_pc" << std::endl;

    // initial field
    printf("Computing initial velocity field\n");

    NSBase.usolver.setMaxIterations(NSBase.maxitsolver);
    NSBase.vsolver.setMaxIterations(NSBase.maxitsolver);
    NSBase.wsolver.setMaxIterations(NSBase.maxitsolver);
    NSBase.alpha = NSBase.alpha0;

    NSBase.AssembleAllMomentum(uA, vA, wA, u, v, w);
    NSBase.ComputeVelocityField(uA, vA, wA, u, v, w, pvector, false, 0);
    // NSBase.ComputePermeability(Q, v_moy, K, u, pvector); // removed: redundant (recomputed on the next line)
    // temp.resize(0); // removed: 'temp' was unused
    NSBase.ComputePermeability(Q, v_moy, K, u, pvector);
    printf("Initial Permeability K (m²): %g\n", K);
    NSBase.info_file << "Initial Permeability K (m²): " << K << std::endl;

    for (it = 0; it < maxit; ++it)
    {
        printf("----------------------------------\n");
        printf("Iteration %d  ---------------------\n", it);
        NSBase.info_file << "Iteration " << it << "  ------------" << std::endl;

        // compute the velocity field using P_n-1
        if (it > 0)
        {
            NSBase.AssembleAllMomentum(uA, vA, wA, u, v, w);
            NSBase.ComputeVelocityField(uA, vA, wA, u, v, w, pvector, true, v_moy * std::min(1., NSBase.alpha * 1.5));
        }
        NSBase.ComputePermeability(Q, v_moy, K, u, pvector);
        NSBase.ComputeDiv(MaxDeltaV, normdiv, velocityNorm, maxdiv, u, v, w);
        NSBase.RMR = MaxDeltaV / std::abs(v_moy);
        reldiv = (NSBase.nx * NSBase.dx + NSBase.ny * NSBase.dy + NSBase.nz * NSBase.dz) * maxdiv / (3 * v_moy);

        printf("Permeability K (m²): %g\n", K);
        NSBase.info_file << "Permeability K (m²): " << K << std::endl;
        printf("Maximum value of the velocity field divergence : %g\n", maxdiv);
        printf("Relative Mass Residual, Lref*(max DIV)/<u_out>, ||Div u|| %g, %g, %g\n", NSBase.RMR, reldiv, normdiv);

        // convergence control
        if (it > 1)
        {
            if (NSBase.RMR < NSBase.old_RMR)
            {
                // counter += 1; // removed: 'counter' was never read
                NSBase.old_RMR = NSBase.RMR;
            }
            // else { counter = 0; } // removed: 'counter' was never read
        }
        else
        {
            NSBase.old_RMR = NSBase.RMR;
        }

        NSBase.log_file << it << ',' << Q << ',' << v_moy << ',' << NSBase.RMR << ',' << K << ',' << (NSBase.p_inlet - NSBase.p_outlet) / (NSBase.dx * (NSBase.pindex.shape()(0) - 2 * NSBase.border)) << ',' << normdiv << ',' << reldiv << ',' << velocityNorm << ", " << maxdiv << ", " << NSBase.usolver.iterations() << ',' << NSBase.vsolver.iterations() << ',' << NSBase.wsolver.iterations() << ',' << NSBase.psolver.iterations() << std::endl;

        // convergence check
        if (it > 1)
        {
            convergence = NSBase.CheckConvergence(NSBase.RMR, simpletol);
            if (convergence)
            {
                printf("Permeability K (m²): %g\n", K);
                NSBase.info_file << "Permeability K (m²): " << K << std::endl;
                break;
            }
        }

        // correct the velocity and pressure

        // solve for P'
        NSBase.AssemblePressureCorrectionSIMPLEC(pA, uA, vA, wA, pvector, u, v, w, b);
        NSBase.ComputePressureCorrection(pc, b);

        NSBase.CorrectVelocityFieldSIMPLEC(pc, u, v, w, uA, vA, wA, v_moy * NSBase.alpha);
        NSBase.CorrectPressureField(pc, pvector);

        NSBase.ComputeDiv(MaxDeltaV, normdiv, velocityNorm, maxdiv, u, v, w);
        printf("Max div after correction: %g\n", maxdiv);

        // save intermediate fields
        if (it % save_step == 0 && it > 0)
        {
            savecounter += 1;
            if (savecounter > 3)
            {
                savecounter = 1;
            }
            savefile = NSBase.outfilename.substr(0, NSBase.outfilename.length() - 4) + "-SAVE-" + std::to_string(savecounter) + ".vti";
            NSBase.FieldsToImage(savefile, pvector, u, v, w);
        }
        NSBase.RelaxControl(true);
    }// end of main loop

    NSBase.info_file << "Norm of the velocity field divergence: " << normdiv << std::endl;
    printf("Norm of the velocity field divergence: %g\n", normdiv);
    NSBase.info_file << "Relative divergence of the velocity field: " << reldiv << std::endl;
    printf("Relative divergence of the velocity field: %g\n", reldiv);

    // delete the matrices to free memory
    uA.resize(0, 0);
    vA.resize(0, 0);
    wA.resize(0, 0);

    // allocate the fields in 3D images for output
    outfilename = NSBase.outfilename.substr(0, NSBase.outfilename.length() - 4) + ".vti";

    NSBase.FieldsToImage(outfilename, pvector, u, v, w);

    end = time(0);

    printf("SIMPLEC terminated in %d iteration and %g seconds\n", it, difftime(end, start));
    NSBase.info_file << "SIMPLEC terminated in " << it << " iteration and " << difftime(end, start) << " seconds" << std::endl;
}
