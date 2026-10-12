/*
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
 *
 *
 *
 * Compile with  /usr/include/eigen3/ -I /usr/include/blitz/ -fopenmp
 *
 *
 *
 *
 *
 */

#define NDEBUG
#include <assert.h>

//OpenMP pas nécessaire ?
#include <omp.h>

//Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#define EIGEN_NO_DEBUG

//Blitz for 3D arrays
#include <blitz/array.h>

//std headers
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <stdio.h>

//project header
#include "SIMPLERR.h"
#include "NSBase.h"


SIMPLERRSOLVER::SIMPLERRSOLVER(double simpletol, int maxit, int N1) :
                        simpletol(simpletol), maxit(maxit), N1(N1) {

    start=time(0);

}


void SIMPLERRSOLVER::Solve(NavierStokesBase & NSBase){

    //~ Initialisation
    bool convergence, relax=true;
    double Q=0, v_moy=0, K=0;
    double normdiv,reldiv,velocityNorm,maxdiv,MaxDeltaV;

    std::string outfilename;
    int nu=NSBase.nu, nv=NSBase.nv, nw=NSBase.nw, np=NSBase.np;

    int it,sub1,N0=20,counter=0;
    std::string savefile;

    save_step = 20;

    std::cout<<"Initializing SIMPLER-R algorithm"<<std::endl;
    NSBase.info_file<<"Initializing SIMPLER-R algorithm"<<std::endl;

    //Allocation du champ de vitesse et pression
    //new velocities
    u.resize(nu);
    u = Eigen::VectorXd::Zero(nu);
    v.resize(nv);
    v = Eigen::VectorXd::Zero(nv);
    w.resize(nw);
    w = Eigen::VectorXd::Zero(nw);

    //pseudo velocities
    u_hat.resize(nu);
    u_hat = Eigen::VectorXd::Zero(nu);
    v_hat.resize(nv);
    v_hat = Eigen::VectorXd::Zero(nv);
    w_hat.resize(nw);
    w_hat = Eigen::VectorXd::Zero(nw);

    u_temp.resize(nu);
    u_temp = Eigen::VectorXd::Zero(nu);
    v_temp.resize(nv);
    v_temp = Eigen::VectorXd::Zero(nv);
    w_temp.resize(nw);
    w_temp = Eigen::VectorXd::Zero(nw);

    Eigen::VectorXd temp;

    pvector.resize(np);
    pvector = Eigen::VectorXd::Zero(np);
    pc.resize(np);
    pc = Eigen::VectorXd::Zero(np);

    NSBase.Initpvector(NSBase.pfieldfilename,pvector);

    //Paramétrage des solveurs
    NSBase.psolver.setMaxIterations(NSBase.maxitsolver);
    NSBase.psolver.setTolerance(NSBase.psolvertol);

    NSBase.usolver.setTolerance(NSBase.vsolvertol);
    NSBase.vsolver.setTolerance(NSBase.vsolvertol);
    NSBase.wsolver.setTolerance(NSBase.vsolvertol);

    NSBase.info_file<<"Max number of SIMPLE-R iterations: "<<maxit<<std::endl;
    std::cout<<"Stopping criterion for relative residual: "<<simpletol<<std::endl;

    NSBase.log_file<<"iteration, Q (m³/s), <u_out> (m/s), Max Rel Mass Res., K (m²), PC_max (Pa), PC_min (Pa) ||Div_u||, Lref*||Div u||/<||u||>, <||u||>, max DIV, it_u,it_v,it_w,it_pc"<<std::endl;

    //Champ initial
    std::cout<<"Computing initial velocity field"<<std::endl;
    if (NSBase.pfieldfilename != "none"){

        NSBase.usolver.setMaxIterations(100);
        NSBase.vsolver.setMaxIterations(100);
        NSBase.wsolver.setMaxIterations(100);

        for (it=0;it<N0;it++){
            std::cout<<"Iteration "<<it<<std::endl;
            NSBase.AssembleAllMomentum(uA,vA,wA,u,v,w);
            NSBase.ComputeVelocityField(uA,vA,wA,u,v,w,pvector);
        }
    }else{
            NSBase.AssembleMomentum_x(uA, NSBase.uindex, u, v, w);
            NSBase.usolver.compute(uA);
            NSBase.Assemble_RHS_u(uA,pvector,u,b);
            temp.resize(nu);
            temp = u;
            u=NSBase.usolver.solveWithGuess(b,u);
            if (NSBase.usolver.iterations() == 0){
                u = temp;
            }
            std::cout<<"Solving for component u in "<<NSBase.usolver.iterations()<<" itérations"<<" (status "<< NSBase.usolver.info()<< ")"<<std::endl;
            NSBase.info_file<<"Solving for component u in "<<NSBase.usolver.iterations()<<" itérations"<<" (status "<< NSBase.usolver.info()<< ")"<<std::endl;
    }

    temp.resize(0);
    NSBase.ComputePermeability(Q, v_moy, K,u);
    std::cout<<"Initial Permeability K (m²): "<<K <<std::endl;
    NSBase.info_file<<"Initial Permeability K (m²): "<<K <<std::endl;

    NSBase.usolver.setMaxIterations(NSBase.maxitsolver);
    NSBase.vsolver.setMaxIterations(NSBase.maxitsolver);
    NSBase.wsolver.setMaxIterations(NSBase.maxitsolver);
    NSBase.alpha=NSBase.alpha0;

    for (it=0;it<maxit;++it){

        std::cout<<"----------------------------------"<<std::endl;
        std::cout<<"Iteration "<< it<< "  ---------------------"<<std::endl;
        NSBase.info_file<<"Iteration "<< it<< "  ------------"<<std::endl;

        //Assemblage des matrices
        NSBase.AssembleAllMomentum(uA,vA,wA,u,v,w);

        u_temp = u;
        v_temp = v;
        w_temp = w;

        //Step 1: compute pressure field P* from old velocity field

        std::cout<<"STEP 1 -- Updating pressure field"<<std::endl;

        for (sub1 = 0; sub1<N1;sub1++){

            //Calcul des pseudos vitesses et des coeffs Au, Av et Aw

            std::cout<<"iteration "<<sub1<<" ";
            NSBase.Compute_u_hat(u_hat, uA, u_temp);
            NSBase.Compute_v_hat(v_hat, vA, v_temp);
            NSBase.Compute_w_hat(w_hat, wA, w_temp);

            //Solving for P*
            NSBase.AssemblePressure(relax,pA,uA,vA,wA,pvector,u_hat,v_hat,w_hat,b);
            NSBase.ComputePressureField(pvector, b);
            NSBase.CorrectVelocityFieldSIMPLER(pvector, u_temp, v_temp, w_temp, u_hat, v_hat, w_hat, uA, vA, wA);

        }

        //Compute u*,v*,w* fields using u_n-1,v_n-1,w_n-1 and P*
        std::cout<<"STEP 2 -- Updating velocity field "<<std::endl;
        NSBase.ComputeVelocityField(uA, vA, wA, u, v, w, pvector);
        NSBase.ComputePermeability(Q, v_moy, K, u);
        NSBase.ComputeDiv(MaxDeltaV,normdiv,velocityNorm,maxdiv,u,v,w);
        NSBase.RMR = MaxDeltaV/std::abs(v_moy);
        reldiv=(NSBase.nx*NSBase.dx+NSBase.ny*NSBase.dy+NSBase.nz*NSBase.dz)*normdiv/(3*velocityNorm);

        std::cout<<"Permeability K (m²): "<<K <<std::endl;
        NSBase.info_file<<"Permeability K (m²): "<<K <<std::endl;
        std::cout<<"Maximum value of the velocity field divergence : "<<maxdiv<<std::endl;
        std::cout<<"Relative Mass Residual, Lref*||Div u||/<||u||>,  ||Div u|| "<<NSBase.RMR<<','<<reldiv<<", "<<normdiv<<std::endl;

        //Convergence control
        if (it>1){

            if (NSBase.RMR<NSBase.old_RMR){
                counter+=1;
                NSBase.old_RMR=NSBase.RMR;
            }else{
                counter=0;
            }
        }else{
            NSBase.old_RMR = NSBase.RMR;
        }



        // Contrôle de la convergence
        if (it>1){
            convergence=NSBase.CheckConvergence(NSBase.RMR,simpletol);
            if(convergence){
                std::cout<<"Permeability K (m²): "<<K <<std::endl;
                NSBase.info_file<<"Permeability K (m²): "<<K <<std::endl;
                break;
            }
        }

        //Compute final velocity and pressure
        std::cout<<"STEP 3 -- Updating pressure & velocity field"<<std::endl;

        //Solving for P'
        NSBase.AssemblePressureCorrectionSIMPLE(pA,uA,vA,wA,pvector,u,v,w,b);
        NSBase.ComputePressureCorrection(pc,b);
        //Phase de correction de la vitesse de type SIMPLE
        NSBase.CorrectVelocityFieldSIMPLE(pc,u,v,w,uA,vA,wA);

        NSBase.Compute_u_hat(u_hat, uA, u);
        NSBase.Compute_v_hat(v_hat, vA, v);
        NSBase.Compute_w_hat(w_hat, wA, w);

        //Mise à jour de la pression
        NSBase.AssemblePressure(relax,pA,uA,vA,wA,pvector,u_hat,v_hat,w_hat,b);
        NSBase.ComputePressureField(pvector, b);

        NSBase.ComputeDiv(MaxDeltaV,normdiv,velocityNorm,maxdiv,u,v,w);
        std::cout<<"Max div after correction:"<<maxdiv<<std::endl;

        NSBase.log_file<<it<<','<< Q<<','<< v_moy<<','<<NSBase.RMR<<','<< K<<','<< NSBase.pcmax<<','<<NSBase.pcmin <<',' << normdiv<<','<<reldiv<<','<<velocityNorm<<", "<<maxdiv<<", "<<NSBase.usolver.iterations()<<','<<NSBase.vsolver.iterations()<<','<<NSBase.wsolver.iterations()<<','<<NSBase.psolver.iterations()<<std::endl;

        NSBase.RelaxControl();

        //Save  intermediate fields
        if(it%save_step == 0 && it>0){
            savefile=NSBase.outfilename.substr(0,NSBase.outfilename.length()-4)+"-"+std::to_string(it)+".vti";
            NSBase.FieldsToImage(savefile,pvector,u,v,w);
        }

    }//end of main loop

    NSBase.info_file<<"Norme de la Divergence du champ de vitesse: "<< normdiv <<std::endl;
    std::cout<<"Norme de la Divergence du champ de vitesse: "<< normdiv <<std::endl;
    NSBase.info_file<<"Divergence relative du champ de vitesse: "<< reldiv <<std::endl;
    std::cout<<"Divergence relative du champ de vitesse: "<< reldiv <<std::endl;

    //Delete the matrices to free memory
    uA.resize(0,0);
    vA.resize(0,0);
    wA.resize(0,0);

    //Allocate the fields in 3D images for output

    outfilename=NSBase.outfilename.substr(0,outfilename.length()-4)+".vti";

    NSBase.FieldsToImage(outfilename,pvector,u,v,w);

    end=time(0);

    std::cout<<"SIMPLER-R terminated in "<< it<< " iteration and "<< difftime(end,start)<<" seconds"<<std::endl;
    NSBase.info_file<<"SIMPLER-R terminated in "<< it<< " iteration and "<< difftime(end,start)<<" seconds"<<std::endl;


}



