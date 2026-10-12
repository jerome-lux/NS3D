/*
 * NSBase.h
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

#ifndef NSBASE_H
#define NSBASE_H

// Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

// Blitz++ for 3D arrays
#include <blitz/array.h>

// std headers
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <list>

/**
 * @brief Base class of all solvers: steady, incompressible, low-Reynolds
 *        Navier-Stokes flow in a porous medium described by a binary 3D
 *        image (0 = fluid, 1 = solid).
 *
 * The class stores the geometry (pressure / velocity index arrays),
 * assembles the sparse linear systems (momentum and pressure equations),
 * drives the iterative linear solvers (Eigen BiCGSTAB), computes the
 * diagnostics (divergence, Darcy permeability), manages the adaptive
 * relaxation parameters and reads/writes the fields.
 *
 * The concrete algorithms (SIMPLEC, SIMPLER, SIMPLERR, IDEAL) are
 * separate classes that call the methods of this class in a different
 * order.
 *
 * The flow is always solved along the x direction (inlet at x = 0,
 * outlet at x = nx-1). If the imposed pressure gradient is along y or z,
 * the image is first rotated with Rotation().
 *
 * All methods that build matrices or loop over the grid are
 * parallelized with OpenMP when available.
 */
class NavierStokesBase
{
public:

    /* ------------------------------------------------------------------ */
    /* Input parameters (set by the constructor)                          */
    /* ------------------------------------------------------------------ */
    int maxitsolver, porevalue, gradient_direction, save_step, border;
    double dx, dy, dz, gradp, alpha0, beta0, alpha, beta, mu, rho, p1, p0, p_outlet, p_inlet;
    double vsolvertol, psolvertol, pcsolvertol, deltaRMR;

    /* ------------------------------------------------------------------ */
    /* Global state                                                       */
    /* ------------------------------------------------------------------ */
    int nx, ny, nz, np, nu, nv, nw;
    int INCRELAXCOUNTER;
    double schemefactor;
    double pcmax, pcmin, pcnorm;
    double min_alpha, min_beta;
    double relax, maxconvdelta, RMR, old_RMR, speedup;
    std::list<double> avgRMR;

    // Binary image of the microstructure (0 = fluid, 1 = solid), freed after initialization
    blitz::Array<unsigned char, 3> image;

    // Index maps: spatial coordinates -> entries of the linear systems.
    // A value of -1 means "no node of that field at this location".
    blitz::Array<int, 3> pindex;
    blitz::Array<int, 3> uindex;
    blitz::Array<int, 3> vindex;
    blitz::Array<int, 3> windex;

    // Iterative linear solvers (BiCGSTAB) for the three momentum
    // components and for the pressure equation
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor> > usolver;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor> > vsolver;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor> > wsolver;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor> > psolver;

    std::string outfilename;
    std::string imagefilename;
    std::string info_filename, log_filename, pfieldfilename, vxfilename, vyfilename, vzfilename;
    std::ofstream info_file, log_file;

    /* ------------------------------------------------------------------ */
    /* Member functions                                                   */
    /* ------------------------------------------------------------------ */

    /**
     * @brief Constructor: reads the binary image, builds the index
     *        arrays and prepares the output files.
     *
     * Loads the image with VTK, rotates it so that the flow direction
     * is x (if requested), optionally pads it with a solid border,
     * builds the pressure/velocity index arrays, computes the outlet
     * pressure p1 from the imposed gradient and opens the INFO/RES
     * log files.
     *
     * @param imagefilename    binary image file (0 = fluid, 1 = solid).
     * @param pfieldfilename   initial pressure field file, or "none".
     * @param vxfilename       initial x-velocity field file, or "none".
     * @param vyfilename       initial y-velocity field file, or "none".
     * @param vzfilename       initial z-velocity field file, or "none".
     * @param dx               grid spacing along x (m).
     * @param dy               grid spacing along y (m).
     * @param dz               grid spacing along z (m).
     * @param vsolvertol       velocity solver tolerance.
     * @param psolvertol       pressure solver tolerance.
     * @param pcsolvertol      pressure correction solver tolerance.
     * @param maxitsolver      maximum iterations of the linear solvers.
     * @param alpha            velocity relaxation parameter (stored in alpha0).
     * @param beta             pressure relaxation parameter (stored in beta0).
     * @param p0               inlet pressure (Pa).
     * @param gradp            imposed pressure gradient (Pa/m).
     * @param mu               dynamic viscosity (Pa.s).
     * @param rho              density (kg/m^3).
     * @param porevalue        voxel value that represents the fluid (0 or 1).
     * @param gradient_direction 1 (x), 2 (y) or 3 (z).
     * @param deltaRMR         oscillation detection threshold.
     * @param border           solid border thickness added along x (voxels).
     */
    NavierStokesBase(
            std::string & imagefilename,
            std::string & pfieldfilename, std::string &vxfilename, std::string &vyfilename, std::string &vzfilename,
            double dx, double dy, double dz,
            double vsolvertol, double psolvertol, double pcsolvertol,
            int  maxitsolver,
            double alpha, double beta,
            double p0, double gradp,
            double mu, double rho,
            int porevalue, int gradient_direction,
            double deltaRMR,
            int border);

    /**
     * @brief Assemble the three momentum matrices and factor the systems.
     *
     * Calls AssembleMomentum_x/y/z and runs usolver/vsolver/wsolver
     * compute() on the resulting matrices.
     *
     * @param uA  [out] x-momentum matrix (nu x nu).
     * @param vA  [out] y-momentum matrix (nv x nv).
     * @param wA  [out] z-momentum matrix (nw x nw, skipped if nw == 0).
     * @param uc  current x-velocity (used for the convection terms).
     * @param vc  current y-velocity.
     * @param wc  current z-velocity.
     */
    void AssembleAllMomentum(
                    Eigen::SparseMatrix<double, Eigen::RowMajor> & uA,
                    Eigen::SparseMatrix<double, Eigen::RowMajor> & vA,
                    Eigen::SparseMatrix<double, Eigen::RowMajor> & wA,
                    Eigen::VectorXd &uc,
                    Eigen::VectorXd &vc,
                    Eigen::VectorXd &wc);

    /**
     * @brief Assemble the x-momentum matrix.
     *
     * Finite-volume assembly on the staggered grid: upwind convection
     * if the local Peclet number exceeds 2, centered otherwise; no-slip
     * at solid walls; zero-gradient boundary at the inlet/outlet faces.
     * The diagonal is divided by the relaxation parameter alpha.
     *
     * @param A       [out] matrix to fill (nu x nu).
     * @param cindex  index map of the x-velocity nodes.
     * @param uc      current x-velocity (convection).
     * @param vc      current y-velocity (convection).
     * @param wc      current z-velocity (convection).
     */
    void AssembleMomentum_x(Eigen::SparseMatrix<double, Eigen::RowMajor> & A, const blitz::Array<int, 3> & cindex, Eigen::VectorXd &uc, Eigen::VectorXd &vc, Eigen::VectorXd &wc);

    /**
     * @brief Assemble the y-momentum matrix (see AssembleMomentum_x).
     *
     * @param A       [out] matrix to fill (nv x nv).
     * @param cindex  index map of the y-velocity nodes.
     * @param uc      current x-velocity (convection).
     * @param vc      current y-velocity.
     * @param wc      current z-velocity (convection).
     */
    void AssembleMomentum_y(Eigen::SparseMatrix<double, Eigen::RowMajor> & A, const blitz::Array<int, 3> & cindex, Eigen::VectorXd &uc, Eigen::VectorXd &vc, Eigen::VectorXd &wc);

    /**
     * @brief Assemble the z-momentum matrix (see AssembleMomentum_x).
     *
     * @param A       [out] matrix to fill (nw x nw).
     * @param cindex  index map of the z-velocity nodes.
     * @param uc      current x-velocity (convection).
     * @param vc      current y-velocity (convection).
     * @param wc      current z-velocity.
     */
    void AssembleMomentum_z(Eigen::SparseMatrix<double, Eigen::RowMajor> & A, const blitz::Array<int, 3> & cindex, Eigen::VectorXd &uc, Eigen::VectorXd &vc, Eigen::VectorXd &wc);

    /**
     * @brief Assemble the pressure equation (continuity) matrix.
     *
     * Builds the pressure-Poisson system with the inlet pressure p0 and
     * outlet pressure p1 imposed on the boundary, and optionally applies
     * pressure under-relaxation with beta. Finally factorizes the
     * system with psolver.
     *
     * @param relax   apply pressure under-relaxation (beta).
     * @param pA      [out] pressure matrix (np x np).
     * @param uA,vA,wA momentum matrices (diagonals used).
     * @param pvector current pressure field.
     * @param uc,vc,wc current (or pseudo) velocity fields.
     * @param b       [out] right-hand side.
     */
    void AssemblePressure(const bool &relax,
                    Eigen::SparseMatrix<double, Eigen::RowMajor> & pA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & uA, const Eigen::SparseMatrix<double, Eigen::RowMajor> & vA, const Eigen::SparseMatrix<double, Eigen::RowMajor> & wA,
                    const Eigen::VectorXd &pvector, const Eigen::VectorXd &uc, const Eigen::VectorXd &vc, const Eigen::VectorXd &wc,
                    Eigen::VectorXd &b);

    /**
     * @brief Assemble the SIMPLEC pressure correction matrix.
     *
     * Uses the row sum of the momentum matrices instead of the
     * diagonal (SIMPLEC formulation: no explicit source term for the
     * convection in the correction equation).
     *
     * @param pA      [out] pressure correction matrix (np x np).
     * @param uA,vA,wA momentum matrices.
     * @param pvector current pressure field (unused, kept for API compatibility).
     * @param u,v,w   current velocity fields.
     * @param b       [out] right-hand side.
     */
    void AssemblePressureCorrectionSIMPLEC(
                    Eigen::SparseMatrix<double, Eigen::RowMajor> & pA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & uA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & vA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & wA,
                    const Eigen::VectorXd &pvector, const Eigen::VectorXd &u, const Eigen::VectorXd &v, const Eigen::VectorXd &w,
                    Eigen::VectorXd &b);

    /**
     * @brief Assemble the SIMPLE pressure correction matrix.
     *
     * Uses the diagonal of the momentum matrices (classic SIMPLE
     * formulation with the convection term moved to the right-hand
     * side).
     *
     * @param pA      [out] pressure correction matrix (np x np).
     * @param uA,vA,wA momentum matrices.
     * @param pvector current pressure field (unused, kept for API compatibility).
     * @param uc,vc,wc current velocity fields.
     * @param b       [out] right-hand side.
     */
    void AssemblePressureCorrectionSIMPLE(
                    Eigen::SparseMatrix<double, Eigen::RowMajor> & pA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & uA, const Eigen::SparseMatrix<double, Eigen::RowMajor> & vA, const Eigen::SparseMatrix<double, Eigen::RowMajor> & wA,
                    const Eigen::VectorXd &pvector, const Eigen::VectorXd &uc, const Eigen::VectorXd &vc, const Eigen::VectorXd &wc,
                    Eigen::VectorXd &b);

    /**
     * @brief Build the right-hand side of the x-momentum equation.
     *
     * b = pressure gradient term + relaxed previous velocity
     * (u * a_ii * (1 - alpha)).
     *
     * @param uA      x-momentum matrix (diagonal used for relaxation).
     * @param pvector current pressure field.
     * @param u       previous x-velocity.
     * @param b       [out] right-hand side (nu).
     */
    void Assemble_RHS_u(const Eigen::SparseMatrix<double, Eigen::RowMajor> & uA, const Eigen::VectorXd &pvector, const Eigen::VectorXd &u, Eigen::VectorXd &b);

    /**
     * @brief Build the right-hand side of the y-momentum equation.
     *        (see Assemble_RHS_u)
     *
     * @param vA      y-momentum matrix (diagonal used for relaxation).
     * @param pvector current pressure field.
     * @param v       previous y-velocity.
     * @param b       [out] right-hand side (nv).
     */
    void Assemble_RHS_v(const Eigen::SparseMatrix<double, Eigen::RowMajor> & vA, const Eigen::VectorXd &pvector, Eigen::VectorXd &v, Eigen::VectorXd &b);

    /**
     * @brief Build the right-hand side of the z-momentum equation.
     *        (see Assemble_RHS_u)
     *
     * @param wA      z-momentum matrix (diagonal used for relaxation).
     * @param pvector current pressure field.
     * @param w       previous z-velocity.
     * @param b       [out] right-hand side (nw).
     */
    void Assemble_RHS_w(const Eigen::SparseMatrix<double, Eigen::RowMajor> & wA, const Eigen::VectorXd &pvector, Eigen::VectorXd &w, Eigen::VectorXd &b);

    /**
     * @brief Compute divergence diagnostics of the velocity field.
     *
     * @param MaxDeltaV [out] sum of the absolute face velocities (max over cells).
     * @param normdiv   [out] L2 norm of the cell-wise divergence.
     * @param velocityNorm [out] average velocity magnitude over the fluid nodes.
     * @param maxdiv    [out] maximum absolute cell-wise divergence.
     * @param u,v,w     velocity fields.
     */
    void ComputeDiv(double & MaxDeltaV, double & normdiv, double & velocityNorm, double & maxdiv, const Eigen::VectorXd &u, const Eigen::VectorXd &v, const Eigen::VectorXd &w);

    /**
     * @brief Compute the Darcy permeability from the outlet flux.
     *
     * Averages the outlet velocity, evaluates the inlet/outlet mean
     * pressures (from the imposed values or from the field when a
     * border is used), and computes K = v_moy * mu * L / (p_in - p_out).
     *
     * @param Q       [out] volumetric flow rate (m^3/s).
     * @param v_moy   [out] mean outlet velocity (m/s).
     * @param K       [out] Darcy permeability (m^2).
     * @param u       x-velocity field.
     * @param pvector pressure field.
     */
    void ComputePermeability(double &Q, double &v_moy, double &K, const Eigen::VectorXd &u, const Eigen::VectorXd & pvector);

    /**
     * @brief Compute the pseudo-velocity u_hat (SIMPLER formulation).
     *
     * u_hat = (b - A_off * u_old) / a_ii + u_old * (1 - alpha),
     * i.e. the explicit part of the x-momentum equation with the
     * previous velocity as guess.
     *
     * @param u_hat [out] pseudo-velocity (nu).
     * @param A     x-momentum matrix.
     * @param uc    current x-velocity (off-diagonal products).
     * @param u_old previous x-velocity (relaxation).
     */
    void Compute_u_hat(Eigen::VectorXd &u_hat,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> &A,
                    const Eigen::VectorXd &uc,
                    const Eigen::VectorXd &u_old);

    /**
     * @brief Compute the pseudo-velocity v_hat (see Compute_u_hat).
     *
     * @param v_hat [out] pseudo-velocity (nv).
     * @param A     y-momentum matrix.
     * @param vc    current y-velocity (off-diagonal products).
     * @param v_old previous y-velocity (relaxation).
     */
    void Compute_v_hat(Eigen::VectorXd &v_hat,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> &A,
                    const Eigen::VectorXd &vc,
                    const Eigen::VectorXd &v_old);

    /**
     * @brief Compute the pseudo-velocity w_hat (see Compute_u_hat).
     *
     * @param w_hat [out] pseudo-velocity (nw).
     * @param A     z-momentum matrix.
     * @param wc    current z-velocity (off-diagonal products).
     * @param w_old previous z-velocity (relaxation).
     */
    void Compute_w_hat(Eigen::VectorXd &w_hat,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> &A,
                    const Eigen::VectorXd &wc,
                    const Eigen::VectorXd &w_old);

    /**
     * @brief Solve the three momentum systems and update the velocity.
     *
     * Solves with BiCGSTAB using the current field as initial guess;
     * if a solve fails the previous field is kept. When @p limit is
     * true, every velocity update is clamped to delta_v_max.
     *
     * @param uA,vA,wA momentum matrices (already factored).
     * @param u,v,w    [out] updated velocity fields.
     * @param pvector  current pressure field.
     * @param limit    clamp the velocity update.
     * @param delta_v_max maximum allowed change per velocity node.
     */
    void ComputeVelocityField(
                    Eigen::SparseMatrix<double, Eigen::RowMajor> & uA,
                    Eigen::SparseMatrix<double, Eigen::RowMajor> & vA,
                    Eigen::SparseMatrix<double, Eigen::RowMajor> & wA,
                    Eigen::VectorXd &u,
                    Eigen::VectorXd &v,
                    Eigen::VectorXd &w,
                    const Eigen::VectorXd &pvector,
                    bool limit,
                    double delta_v_max);

    /**
     * @brief Solve the pressure correction system.
     *
     * @param pc [out] pressure correction field (np).
     * @param b  right-hand side.
     */
    void ComputePressureCorrection(Eigen::VectorXd &pc, Eigen::VectorXd &b);

    /**
     * @brief Solve the pressure system.
     *
     * Uses the current field as initial guess and, if the solver
     * converges immediately (zero iterations), tightens the tolerance
     * and retries.
     *
     * @param pvector [out] updated pressure field (np).
     * @param b       right-hand side.
     */
    void ComputePressureField(Eigen::VectorXd &pvector, Eigen::VectorXd &b);

    /**
     * @brief SIMPLE velocity correction: u += -dp/a with clamping.
     *
     * @param pc        pressure correction field.
     * @param u,v,w     [out] corrected velocity fields.
     * @param uA,vA,wA  momentum matrices (diagonals used).
     * @param delta_v_max maximum allowed change per velocity node.
     */
    void CorrectVelocityFieldSIMPLE(
                    const Eigen::VectorXd &pc, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &w,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & uA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & vA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & wA,
                    double delta_v_max);

    /**
     * @brief SIMPLEC velocity correction (uses row sums of the
     *        momentum matrices, see AssemblePressureCorrectionSIMPLEC).
     *
     * @param pc        pressure correction field.
     * @param u,v,w     [out] corrected velocity fields.
     * @param uA,vA,wA  momentum matrices.
     * @param delta_v_max maximum allowed change per velocity node.
     */
    void CorrectVelocityFieldSIMPLEC(const Eigen::VectorXd &pc, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &w,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & uA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & vA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & wA,
                    double delta_v_max);

    /**
     * @brief SIMPLER velocity correction: u = u_hat - dp/a.
     *
     * @param p         pressure field or correction.
     * @param uc,vc,wc  [out] corrected velocity fields.
     * @param u_hat,v_hat,w_hat pseudo-velocities.
     * @param uA,vA,wA  momentum matrices (diagonals used).
     * @param p_0       inlet pressure (0 when p is a correction).
     * @param p_1       outlet pressure (0 when p is a correction).
     */
    void CorrectVelocityFieldSIMPLER(
                    const Eigen::VectorXd &p,
                    Eigen::VectorXd &uc,
                    Eigen::VectorXd &vc,
                    Eigen::VectorXd &wc,
                    const Eigen::VectorXd &u_hat,
                    const Eigen::VectorXd &v_hat,
                    const Eigen::VectorXd &w_hat,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & uA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & vA,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor> & wA,
                    double p_0, double p_1);

    /**
     * @brief Build the pressure/velocity index arrays from the image.
     *
     * A pressure node exists at a fluid cell that has at least one
     * fluid neighbor; velocity nodes are created on the fluid faces
     * (x-faces for u, y-faces for v, z-faces for w). Also sets
     * np, nu, nv, nw.
     *
     * @param image binary image (0 = fluid, 1 = solid).
     */
    void InitIndexArrays(const blitz::Array<unsigned char, 3> image);

    /**
     * @brief Initialize the pressure field.
     *
     * Reads the field from file if @p pfieldfilename is not "none",
     * otherwise builds the linear profile p(x) = p0 + (p1 - p0) * (i+1)/nx.
     *
     * @param pfieldfilename field file name, or "none".
     * @param pvector        [out] pressure field to fill (np).
     */
    void Initpvector(std::string &pfieldfilename, Eigen::VectorXd &pvector);

    /**
     * @brief Initialize the velocity field.
     *
     * Reads the three components from their files if the corresponding
     * filenames are not "none"; otherwise the field stays zero.
     *
     * @param u,v,w [out] velocity fields to fill.
     */
    void InitVelocity(Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &w);

    /**
     * @brief Rotate the image so that the flow direction becomes x.
     *
     * direction = 2: rotate around z (y -> x);
     * direction = 3: rotate around y (z -> x).
     *
     * @param direction 2 or 3.
     * @param image     [out] rotated image.
     */
    void Rotation(int direction, blitz::Array<unsigned char, 3> &image);

    /**
     * @brief Adaptive relaxation control (anti-divergence).
     *
     * Monitors the history of the relative mass residual (RMR): if it
     * grows too much, or if it oscillates / fails to decrease, the
     * relaxation parameters alpha (and optionally beta) are reduced.
     *
     * @param beta_cst if true, beta is kept constant (SIMPLEC/IDEAL
     *                 use it for the momentum only).
     */
    void RelaxControl(bool beta_cst);

    /**
     * @brief Safe vector access.
     *
     * @return vec(indexvector(i,j,k)) if (i,j,k) is inside the array
     *         and a node exists there, 0 otherwise.
     *
     * @param vec          vector of values.
     * @param indexvector  index map of the vector.
     * @param i,j,k        spatial coordinates.
     */
    double SafeVectorValue(const Eigen::VectorXd & vec, const blitz::Array<int, 3> &indexvector, int i, int j, int k);

    /**
     * @brief Apply the pressure correction to the pressure field.
     *
     * p += beta * pc; also stores pcmax/pcmin/pcnorm and prints them.
     *
     * @param pc      pressure correction field.
     * @param pvector [out] updated pressure field.
     */
    void CorrectPressureField(Eigen::VectorXd &pc, Eigen::VectorXd &pvector);

    /**
     * @brief Project the fields onto 3D images and write VTI files.
     *
     * Writes one scalar file per velocity component, the cell-centered
     * (face-averaged) velocity vector and the pressure field, with
     * "Vx-<dir>-", "Vy-<dir>-", "Vz-<dir>-", "V-<dir>-" and
     * "P-<dir>-" prefixed filenames.
     *
     * @param filename base output filename.
     * @param pvector  pressure field.
     * @param u,v,w    velocity fields.
     */
    void FieldsToImage(std::string & filename, Eigen::VectorXd &pvector, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &w);

    /**
     * @brief Convergence test on the relative mass residual.
     *
     * @return true if RMR < tol, or if all the linear solvers reached
     *         a fixed point (zero iterations).
     *
     * @param res current RMR.
     * @param tol convergence threshold.
     */
    bool CheckConvergence(double & res, double & tol);

    /**
     * @brief Pad the domain with a solid border along x.
     *
     * Adds BorderLength solid voxels on both x sides of the image
     * (used to place the inlet/outlet faces inside the domain).
     *
     * @param image        [out] padded image.
     * @param BorderLength thickness of the border (voxels).
     */
    void AddBorder(blitz::Array<unsigned char, 3> &image, int BorderLength);

};

#endif
