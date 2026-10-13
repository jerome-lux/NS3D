/*
 *
 *
 * Copyright 2017 jlux
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
 */

// OpenMP (used by the parallel loops below)
#include <omp.h>

// std headers
#include <cstdio>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>

// Disable the Eigen debug checks: the definition must come before the
// Eigen headers are included to have any effect.
#define EIGEN_NO_DEBUG

// Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

// Blitz++ for 3D arrays
#include <blitz/array.h>

// project headers
#include "NSBase.h"
#include "vtkio.h"

NavierStokesBase::NavierStokesBase(
            std::string & imagefilename,
            std::string & pfieldfilename,
            std::string &vxfilename,
            std::string &vyfilename,
            std::string &vzfilename,
            double dx=1, double dy=1, double dz=1,
            double vsolvertol=1e-5, double psolvertol=1e-6, double pcsolvertol=1e-3,
            int  maxitsolver=500,
            double alpha=1, double beta=1,
            double p0=1e5, double gradp=1e2,
            double mu=1e-3, double rho=1e3,
            int porevalue=0, int gradient_direction=1,double deltaRMR=1e-4, int save_step=20, int border=0) :
            imagefilename(imagefilename),pfieldfilename(pfieldfilename),vxfilename(vxfilename),vyfilename(vyfilename),vzfilename(vzfilename),
            dx(dx),dy(dy),dz(dz),
            vsolvertol(vsolvertol),psolvertol(psolvertol),pcsolvertol(pcsolvertol),
            maxitsolver(maxitsolver),alpha(alpha),beta(beta),p0(p0),gradp(gradp),
            mu(mu),rho(rho),porevalue(porevalue),gradient_direction(gradient_direction),deltaRMR(deltaRMR),save_step(save_step),border(border)
{

    info_file<<"Low Reynolds Steady Navier-Stokes simulation  "<<std::endl;
    info_file<<"Developed by J. Lux 2016, jerome.lux@univ-lr.fr  "<<std::endl;
    info_file<<"LaSIE, University of La Rochelle France "<<std::endl;
    info_file<<"--- Input parameters ---"<<std::endl;
    info_file<<"Image "<<imagefilename.c_str()<<std::endl;
    info_file<<"dx, dy, dz = "<<dx<<","<<dy<<","<<dz<<std::endl;
    info_file<<"viscosity (Pas.s)"<<mu<<std::endl;
    info_file<<"density (kg/m³)"<<rho<<std::endl;
    info_file<<"Relaxation parameter for velocity correction: "<<alpha<<std::endl;
    info_file<<"Relaxation parameter for pressure correction: "<<beta<<std::endl;
    info_file<<"Tolerance for linear system resolution (pressure & velocity & pressure correction): "<<psolvertol<<", "<<vsolvertol<<", "<<pcsolvertol<<std::endl;
    info_file<<"Max number of iteration for the iterative linear solver: "<<maxitsolver<<std::endl;
    info_file<<"Pore voxel value = "<<porevalue<<std::endl;
    info_file<<"Direction of the pressure gradient: "<< gradient_direction<<" (1:x, 2:y, 3:z)"<<std::endl;
    info_file<<"Pressure gradient (Pa/m): "<< gradp <<std::endl;
    info_file<<"Save step (snapshot every N outer iterations): "<< save_step <<std::endl;
    info_file<<"Border (x2): "<< border <<std::endl;
    printf("Low Reynolds Steady Navier-Stokes simulation\n");
    printf("Developed by J. Lux 2016, jerome.lux@univ-lr.fr\n");
    printf("LaSIE, University of La Rochelle France\n");
    printf("--- Input parameters ---\n");
    printf("Image %s\n", imagefilename.c_str());
    printf("dx, dy, dz = %g, %g, %g\n", dx, dy, dz);
    printf("viscosity (Pas.s) = %g\n", mu);
    printf("density (kg/m^3) = %g\n", rho);
    printf("Relaxation parameter for velocity correction: %g\n", alpha);
    printf("Relaxation parameter for pressure correction: %g\n", beta);
    printf("Tolerance for linear system resolution (pressure & velocity & pressure correction): %g, %g, %g\n", psolvertol, vsolvertol, pcsolvertol);
    printf("Max number of iteration for the iterative linear solver: %d\n", maxitsolver);
    printf("Pore voxel value = %d\n", porevalue);
    printf("Direction of the pressure gradient: %d (1:x, 2:y, 3:z)\n", gradient_direction);
    printf("Pressure gradient (Pa/m): %g\n", gradp);
    printf("Save step (snapshot every N outer iterations): %d\n", save_step);
    printf("Border (x2): %d\n", border);

    /* Retrieve the binary image, create the index arrays and initialize the pressure field */

    std::string path="//";
    outfilename=imagefilename;

    size_t pathfound=outfilename.find_last_of(path);
    if(pathfound!=std::string::npos){
        outfilename.erase(0,pathfound+1);
    }

    std::stringstream ss,ss1;
    ss<<"INFO-"<<gradient_direction<<"-"<<outfilename.substr(0,outfilename.length()-4)<<".log";
    ss1<<"RES-"<<gradient_direction<<"-"<<outfilename.substr(0,outfilename.length()-4)<<".csv";
    info_filename=ss.str();
    log_filename=ss1.str();
    info_file.open(info_filename.c_str(),std::ios::out);
    log_file.open(log_filename.c_str(),std::ios::out);

    info_file.precision(15);

    //~ ioerr(info_file,info_filename);

    readvtkfile_uint8(imagefilename, porevalue, image);

    // It would be possible to implement a resizeAndPreserve to vary the volume of computation

    if (gradient_direction>1){
        Rotation(gradient_direction, image);
    }
    if (border >0){
        AddBorder(image,border);
    }

    nx=image.shape()(0);
    ny=image.shape()(1);
    nz=image.shape()(2);

    // The grid size is only known once the image has been read
    info_file<<"Size "<<nx<<","<<ny<<","<<nz<<std::endl;
    printf("Size %d, %d, %d\n", nx, ny, nz);

    pindex.resize(image.shape());
    uindex.resize(nx+1,ny,nz);
    vindex.resize(image.shape());
    windex.resize(image.shape());

    InitIndexArrays(image);

    image.free(); // no longer needed

    p1=-gradp*(dx*pindex.shape()(0))+p0;


    if (p1<0){
        p0 -= p1;
        p1 = 0;
    }

    // INTENTIONAL: the convection terms are discretized with the UPWIND
    // scheme whenever there is flow. This is achieved by setting
    // schemefactor = 0, which degenerates the Peclet-number switch test
    //     |F|*dx^2 > 2*mu*schemefactor
    // into   F != 0,
    // so the centered branch is never taken. An always-upwind convection
    // is unconditionally stable (no spurious oscillations) for the
    // low-Reynolds flows solved here, which is the desired behaviour.
    schemefactor = 0;
    min_alpha = 0.001;
    min_beta = 0.001;
    relax = 0.05;
    speedup = relax*0.5;
    maxconvdelta = 2.5;
    alpha0=alpha;
    beta0=beta;
    INCRELAXCOUNTER=5;


    info_file<<"Porosity: "<< float(np)/float(nx*ny*nz)<<std::endl;
    info_file<<"Nodes for the pressure field: " << np<<std::endl;
    info_file<<"Nodes for the u velocity field: " << nu<<std::endl;
    info_file<<"Nodes for the v velocity field: " << nv<<std::endl;
    info_file<<"Nodes for the w velocity field: " << nw<<std::endl;
    info_file<<"Delta RMR min : " << deltaRMR<<std::endl;
    printf("Porosity: %g\n", float(np)/float(nx*ny*nz));
    printf("Nodes for the pressure field: %d\n", np);
    printf("Nodes for the u velocity field: %d\n", nu);
    printf("Nodes for the v velocity field: %d\n", nv);
    printf("Nodes for the w velocity field: %d\n", nw);
    printf("Delta RMR min : %g\n", deltaRMR);
}

void NavierStokesBase::RelaxControl(bool beta_cst){


    std::list<double>::iterator it;
    int i,n1,n2;
    double avg1=0,avg2=0;
    double delta;
    const int size=30;
    bool isDecreasing;
    bool freeze = false ;

    //To avoid Divergence
    if(RMR>old_RMR*maxconvdelta ){ //)
        alpha=std::max(min_alpha,alpha-relax);
        //beta=std::max(min_beta,beta*relax);
        if (beta_cst == false){
            beta=std::max(min_beta,beta-relax);
        }
        old_RMR*=1.1;
        freeze = true;
        printf("Relative Mass Residual increases too much -> new relaxation parameter alpha, beta: %g, %g\n", alpha, beta);
        info_file<<"Relative Mass Residual increases too much -> new relaxation parameter alpha, beta: "<< alpha<<", "<<beta<<std::endl;
        avgRMR.pop_front();
        //~ pc = Eigen::VectorXd::Zero(np);
    }

    //Convergence control
    if (avgRMR.size() >0){
        if (avgRMR.back() > RMR){
            isDecreasing = true;
        }else{
            isDecreasing = false;
        }
        std::printf("delta RMR = %6.3e \n",RMR-avgRMR.back());
    }
    avgRMR.push_back(RMR);

    if(avgRMR.size()>size) {
        avgRMR.pop_front();
        i=0;
        n1=0;
        n2=0;

        for(it=avgRMR.begin();it!=avgRMR.end();++it){
            i+=1;
            if(i<avgRMR.size()*0.5){
                avg1+=*it;
                n1+=1;
            }
            else if (i>=avgRMR.size()*0.5){
                avg2+=*it;
                n2+=1;
            }
        }
        avg1 /= n1;
        avg2 /= n2;

        delta = (avg2-avg1)/std::max(avg1,avg2);
        std::printf("avg RMRold = %f, avg RMRnew = %f, delta avgRMR / avgRMR_max = %f %%\n",avg1,avg2,delta*100);

        if (delta<=0 && std::abs(delta)<deltaRMR && isDecreasing == false){       //slow convergence
            i=0;
            while(i<15){
                i++;
                avgRMR.pop_front();
            }
            //alpha=std::max(min_alpha,alpha-relax);
            alpha=std::min(alpha0,alpha-relax*0.25);
            if (beta_cst == false){
                beta=std::max(min_beta,beta-relax*0.25);
            }
            printf("Simulation is oscillating -> new relaxation parameter alpha, beta: %g, %g\n", alpha, beta);
            info_file<<"Simulation is oscillating -> new relaxation parameter alpha, beta: "<< alpha<<", "<<beta<<std::endl;

        //}else if (delta<0 && std::abs(delta)>=deltaRMR && isDecreasing == true && freeze == false){
            //if (alpha < alpha0 || beta < beta0){
                //i=0;
                //while(i<5){
                    //i++;
                    //avgRMR.pop_front();
                //}

                //alpha=std::min(alpha0,alpha+speedup);
                //if (beta_cst == false){
                    //beta=std::min(beta0,beta+speedup);
                //}
                //speedup *= 0.8;
                //std::cout<<"Speed-up tentative: Increasing alpha, beta: "<< alpha<<", "<<beta<<std::endl;
                //info_file<<"Speed-up tentative: Increasing alpha, beta: "<< alpha<<", "<<beta<<std::endl;
            //}

        }else if (delta>0 && isDecreasing == false){        // divergence
            i=0;
            while(i<15){
                i++;
                avgRMR.pop_front();
            }
            alpha=std::max(min_alpha,alpha-relax);
            if (beta_cst == false){
                beta=std::max(min_beta,beta-relax);
            }
            printf("Simulation is diverging -> new relaxation parameter alpha, beta: %g, %g\n", alpha, beta);
            info_file<<"Simulation is diverging -> new relaxation parameter alpha, beta: "<< alpha<<", "<<beta<<std::endl;
        }
    }
}


bool NavierStokesBase::CheckConvergence(double & res, double & tol){


    if (res<tol){
        printf("Converged ! RMR=: %g\n", res);
        return true;
    }

    if (nz>0){
        if((usolver.iterations()==0 && vsolver.iterations()==0 && wsolver.iterations()==0) && psolver.iterations()==0){
            printf("Converged !\n");
            printf("Velocity and pressure field stabilized\n");
            return true;
        }
    }else{
        if((usolver.iterations()==0 && vsolver.iterations()==0) && psolver.iterations()==0){
            printf("Converged !\n");
            printf("Velocity and pressure field stabilized\n");
            return true;
        }
    }

    return false;

}

void NavierStokesBase::FieldsToImage(std::string & filename, Eigen::VectorXd &pvector, Eigen::VectorXd &u,Eigen::VectorXd &v,Eigen::VectorXd &w){


    blitz::Array<double,3> pfield(pindex.shape());
    blitz::Array<double,3> ufield(uindex.shape());
    blitz::Array<double,3> vfield(vindex.shape());
    blitz::Array<double,3> wfield(windex.shape());


    std::string pfield_filename, vfield_filename, vxfield_filename, vyfield_filename, vzfield_filename;
    std::stringstream ss;

    ss<<"P-"<<gradient_direction<<"-"<<filename;
    pfield_filename=ss.str();

    ss.str(std::string());
    ss<<"V-"<<gradient_direction<<"-"<<filename;
    vfield_filename=ss.str();

    ss.str(std::string());
    ss<<"Vx-"<<gradient_direction<<"-"<<filename;
    vxfield_filename=ss.str();

    ss.str(std::string());
    ss<<"Vy-"<<gradient_direction<<"-"<<filename;
    vyfield_filename=ss.str();

    ss.str(std::string());
    ss<<"Vz-"<<gradient_direction<<"-"<<filename;
    vzfield_filename=ss.str();

    pfield=0;
    ufield=0;
    vfield=0;
    wfield=0;

    int i,j,k;

    #pragma omp parallel for default(shared) private(i,j,k)

    for (i=0;i<uindex.shape()(0);i++){
        for (j=0;j<uindex.shape()(1);j++){
            for (k=0;k<uindex.shape()(2);k++){

                if (uindex(i,j,k)>-1){
                    ufield(i,j,k)=u(uindex(i,j,k));
                }
            }
        }
    }

    #pragma omp parallel for default(shared) private(i,j,k)
    for (i=0;i<vindex.shape()(0);i++){
        for (j=0;j<vindex.shape()(1);j++){
            for (k=0;k<vindex.shape()(2);k++){

                if (vindex(i,j,k)>-1){
                    vfield(i,j,k)=v(vindex(i,j,k));
                }
            }
        }
    }

    #pragma omp parallel for default(shared) private(i,j,k)
    for (i=0;i<windex.shape()(0);i++){
        for (j=0;j<windex.shape()(1);j++){
            for (k=0;k<windex.shape()(2);k++){

                if (windex(i,j,k)>-1){
                    wfield(i,j,k)=w(windex(i,j,k));
                }
            }
        }
    }

    writeScalarImage(vxfield_filename,ufield);
    writeScalarImage(vyfield_filename,vfield);
    writeScalarImage(vzfield_filename,wfield);

    ufield.free();
    vfield.free();
    wfield.free();

    blitz::Array<double,3> avg_ufield(pindex.shape());
    blitz::Array<double,3> avg_vfield(pindex.shape());
    blitz::Array<double,3> avg_wfield(pindex.shape());

    avg_ufield=0;
    avg_vfield=0;
    avg_wfield=0;

    #pragma omp parallel for default(shared) private(i,j,k)

    //Allocate pressure and velocity field to 3D images

    for (i=0;i<pindex.shape()(0);i++){
        for (j=0;j<pindex.shape()(1);j++){
            for (k=0;k<pindex.shape()(2);k++){

                if (pindex(i,j,k)>-1){
                    pfield(i,j,k)=pvector(pindex(i,j,k));
                }

                if (uindex(i+1,j,k)>-1 && uindex(i,j,k)>-1){
                    avg_ufield(i,j,k)=(u(uindex(i+1,j,k))+u(uindex(i,j,k)))*0.50;
                }else if (uindex(i+1,j,k)>-1){
                    avg_ufield(i,j,k)=(u(uindex(i+1,j,k)))*0.50;
                }else if (uindex(i,j,k)>-1){
                    avg_ufield(i,j,k)=(u(uindex(i,j,k)))*0.50;
                }

                if (j<pindex.shape()(1)-1){
                    if (vindex(i,j+1,k)>-1 && vindex(i,j,k)>-1){
                        avg_vfield(i,j,k)=(v(vindex(i,j+1,k))+v(vindex(i,j,k)))*0.50;
                    }else if (vindex(i,j+1,k)>-1){
                        avg_vfield(i,j,k)=(v(vindex(i,j+1,k)))*0.50;
                    }else if (vindex(i,j,k)>-1){
                        avg_vfield(i,j,k)=(v(vindex(i,j,k)))*0.50;
                    }
                }else if (vindex(i,j,k)>-1){
                    avg_vfield(i,j,k)=(v(vindex(i,j,k)))*0.50;
                }

                if (k<pindex.shape()(2)-1){
                    if (windex(i,j,k+1)>-1 && windex(i,j,k)>-1){
                        avg_wfield(i,j,k)=(w(windex(i,j,k+1))+w(windex(i,j,k)))*0.50;
                    }else if (windex(i,j,k+1)>-1){
                        avg_wfield(i,j,k)=(w(windex(i,j,k+1)))*0.50;
                    }else if (windex(i,j,k)>-1){
                        avg_wfield(i,j,k)=(w(windex(i,j,k)))*0.50;
                    }
                }else if (windex(i,j,k)>-1){
                    avg_wfield(i,j,k)=(w(windex(i,j,k)))*0.50;
                }
            }
        }
    }

    writeScalarImage(pfield_filename,pfield);
    writeVectorImage(vfield_filename,avg_ufield,avg_vfield,avg_wfield);
    info_file<<"Writing "<<vxfield_filename<<std::endl;
    info_file<<"Writing "<<vyfield_filename<<std::endl;
    info_file<<"Writing "<<vzfield_filename<<std::endl;
    info_file<<"Writing "<<vfield_filename<<std::endl;
    info_file<<"Writing "<<pfield_filename<<std::endl;

}

void NavierStokesBase::AssemblePressure(
                        const bool &relax,
                        Eigen::SparseMatrix<double,Eigen::RowMajor> & pA,
                        const Eigen::SparseMatrix<double,Eigen::RowMajor> & uA,
                        const Eigen::SparseMatrix<double,Eigen::RowMajor> & vA,
                        const Eigen::SparseMatrix<double,Eigen::RowMajor> & wA,
                        const Eigen::VectorXd &pvector, const Eigen::VectorXd &uc,
                        const Eigen::VectorXd &vc, const Eigen::VectorXd &wc,
                        Eigen::VectorXd &b){

    /* Linear system associated with the continuity equation */

    //~ std::cout<<"Assembling the linear system matrix for the pressure field...";

    b.resize(pvector.innerSize());
    b = Eigen::VectorXd::Zero(pvector.innerSize());

    pA.resize(pvector.innerSize(),pvector.innerSize());
    pA.reserve(Eigen::VectorXi::Constant(pvector.innerSize(),7));

    double dx2,dy2,dz2,idx,idy,idz;




    int i,j,k;

    dx2=(dx*dx);
    dy2=(dy*dy);
    dz2=(dz*dz);
    idx=1/dx;
    idy=1/dy;
    idz=1/dz;

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k)

    // Traverse the image
    for (i=0;i<pindex.shape()(0);i++){
        for (j=0;j<pindex.shape()(1);j++){
            for (k=0;k<pindex.shape()(2);k++){

                if (pindex(i,j,k)>-1){   // Node found

                    pA.insert(pindex(i,j,k),pindex(i,j,k))=0;

                    // In the bulk
                    if(i>0){                        //direction x-
                        if (pindex(i-1,j,k)>-1 && uindex(i,j,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.coeff(uindex(i,j,k),uindex(i,j,k)) )*dx2);
                            pA.insert(pindex(i,j,k),pindex(i-1,j,k)) = - 1/(( uA.coeff(uindex(i,j,k),uindex(i,j,k)) )*dx2);
                            b(pindex(i,j,k))-=uc(uindex(i,j,k))*idx;

                        }
                    }
                    if (i<pindex.shape()(0)-1){       //direction x+
                        if (pindex(i+1,j,k)>-1 && uindex(i+1,j,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.coeff(uindex(i+1,j,k),uindex(i+1,j,k)) )*dx2);
                            pA.insert(pindex(i,j,k),pindex(i+1,j,k)) = - 1/(( uA.coeff(uindex(i+1,j,k),uindex(i+1,j,k)) )*dx2);
                            b(pindex(i,j,k))+=uc(uindex(i+1,j,k))*idx;
                        }
                    }
                    if(j>0){                        //direction y-
                        if (pindex(i,j-1,k)>-1 && vindex(i,j,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( vA.coeff(vindex(i,j,k),vindex(i,j,k)) )*dy2);
                            pA.insert(pindex(i,j,k),pindex(i,j-1,k)) = - 1/(( vA.coeff(vindex(i,j,k),vindex(i,j,k)) )*dy2);
                            b(pindex(i,j,k))-=vc(vindex(i,j,k))*idy;
                        }
                    }
                    if (j<pindex.shape()(1)-1){       //direction y+
                        if (pindex(i,j+1,k)>-1 && vindex(i,j+1,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( vA.coeff(vindex(i,j+1,k),vindex(i,j+1,k)) )*dy2);
                            pA.insert(pindex(i,j,k),pindex(i,j+1,k)) = - 1/(( vA.coeff(vindex(i,j+1,k),vindex(i,j+1,k)) )*dy2);
                            b(pindex(i,j,k))+=vc(vindex(i,j+1,k))*idy;
                        }
                    }
                    if(k>0 && nz>0){                        //direction z-
                        if (pindex(i,j,k-1)>-1 && windex(i,j,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( wA.coeff(windex(i,j,k),windex(i,j,k)) )*dz2);
                            pA.insert(pindex(i,j,k),pindex(i,j,k-1)) = - 1/(( wA.coeff(windex(i,j,k),windex(i,j,k)) )*dz2);
                            b(pindex(i,j,k))-=wc(windex(i,j,k))*idz;
                        }
                    }
                    if (k<pindex.shape()(2)-1 && nz>0){       //direction z+
                        if (pindex(i,j,k+1)>-1 && windex(i,j,k+1)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/( ( wA.coeff(windex(i,j,k+1),windex(i,j,k+1)) )*dz2);
                            pA.insert(pindex(i,j,k),pindex(i,j,k+1)) = - 1/( ( wA.coeff(windex(i,j,k+1),windex(i,j,k+1)) )*dz2);
                            b(pindex(i,j,k))+=wc(windex(i,j,k+1))*idz;
                        }
                    }

                    // and now the boundary conditions...

                    if (i==0 && uindex(i,j,k)>-1){  //direction x-

                        pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.coeff(uindex(i,j,k),uindex(i,j,k)) )*dx2);
                        b(pindex(i,j,k)) -= uc(uindex(i,j,k))*idx - p0/(uA.coeff(uindex(i,j,k),uindex(i,j,k))*dx2);

                    }
                    if (i==pindex.shape()(0)-1 && uindex(i+1,j,k)>-1){  //direction x+

                        pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.coeff(uindex(i+1,j,k),uindex(i+1,j,k)) )*dx2);
                        b(pindex(i,j,k)) += uc(uindex(i+1,j,k))*idx + p1/( uA.coeff(uindex(i+1,j,k),uindex(i+1,j,k))*dx2);

                    }
                        // Nothing to add for the other directions, as the velocity is zero

                    //under relaxation of pressure equation
                    if (relax){
                        pA.coeffRef(pindex(i,j,k),pindex(i,j,k))/=beta;
                        b(pindex(i,j,k)) += pvector(pindex(i,j,k))*pA.coeffRef(pindex(i,j,k),pindex(i,j,k))*(1-beta);
                    }

                }//end node found

            }//Inner for loop
        }
    }//outer for loop
    pA.makeCompressed();
    psolver.compute(pA);
    if(psolver.info()!=Eigen::Success) {
    // decomposition failed
        printf("Failure decomposing Matrix\n");

    }//else std::cout<<"Matrix decomposition OK. "<<std::endl;

}

void NavierStokesBase::AssemblePressureCorrectionSIMPLEC(
                        Eigen::SparseMatrix<double,Eigen::RowMajor> & pA,
                        const Eigen::SparseMatrix<double,Eigen::RowMajor> & uA,
                        const Eigen::SparseMatrix<double,Eigen::RowMajor> & vA,
                        const Eigen::SparseMatrix<double,Eigen::RowMajor> & wA,
                        const Eigen::VectorXd &pvector,
                        const Eigen::VectorXd &u,
                        const Eigen::VectorXd &v,
                        const Eigen::VectorXd &w,
                        Eigen::VectorXd &b){

    /* SIMPLEC-style pressure correction formulation. */

    printf("Assembling the linear system matrix for the pressure correction...");

    b.resize(pvector.innerSize());
    b = Eigen::VectorXd::Zero(pvector.innerSize());

    pA.resize(pvector.innerSize(),pvector.innerSize());
    pA.reserve(Eigen::VectorXi::Constant(pvector.innerSize(),7));

    double dx2,dy2,dz2,idx,idy,idz;

    int i,j,k;

    dx2=(dx*dx);
    dy2=(dy*dy);
    dz2=(dz*dz);
    idx=1/dx;
    idy=1/dy;
    idz=1/dz;


    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k)

    // Traverse the image
    for (i=0;i<pindex.shape()(0);i++){
        for (j=0;j<pindex.shape()(1);j++){
            for (k=0;k<pindex.shape()(2);k++){

                if (pindex(i,j,k)>-1){   // Node found
                    pA.insert(pindex(i,j,k),pindex(i,j,k))=0;

                    // In the bulk
                    if(i>0){                        //direction x-
                        if (pindex(i-1,j,k)>-1 && uindex(i,j,k)>-1){
                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.row(uindex(i,j,k)).sum() )*dx2);
                            pA.insert(pindex(i,j,k),pindex(i-1,j,k)) = - 1/(( uA.row(uindex(i,j,k)).sum() )*dx2);
                            b(pindex(i,j,k))-=u(uindex(i,j,k))*idx;
                        }
                    }
                    if (i<pindex.shape()(0)-1){       //direction x+
                        if (pindex(i+1,j,k)>-1 && uindex(i+1,j,k)>-1){
                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.row(uindex(i+1,j,k)).sum() )*dx2);
                            pA.insert(pindex(i,j,k),pindex(i+1,j,k)) = - 1/(( uA.row(uindex(i+1,j,k)).sum() )*dx2);
                            b(pindex(i,j,k))+=u(uindex(i+1,j,k))*idx;
                        }
                    }
                    if(j>0){                        //direction y-
                        if (pindex(i,j-1,k)>-1 && vindex(i,j,k)>-1){
                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( vA.row(vindex(i,j,k)).sum() )*dy2);
                            pA.insert(pindex(i,j,k),pindex(i,j-1,k)) = - 1/(( vA.row(vindex(i,j,k)).sum() )*dy2);
                            b(pindex(i,j,k))-=v(vindex(i,j,k))*idy;
                        }
                    }
                    if (j<pindex.shape()(1)-1){       //direction y+
                        if (pindex(i,j+1,k)>-1 && vindex(i,j+1,k)>-1){
                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( vA.row(vindex(i,j+1,k)).sum() )*dy2);
                            pA.insert(pindex(i,j,k),pindex(i,j+1,k)) = - 1/(( vA.row(vindex(i,j+1,k)).sum() )*dy2);
                            b(pindex(i,j,k))+=v(vindex(i,j+1,k))*idy;
                        }
                    }
                    if(k>0 && nz>0){                        //direction z-
                        if (pindex(i,j,k-1)>-1 && windex(i,j,k)>-1){
                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( wA.row(windex(i,j,k)).sum() )*dz2);
                            pA.insert(pindex(i,j,k),pindex(i,j,k-1)) = - 1/(( wA.row(windex(i,j,k)).sum() )*dz2);
                            b(pindex(i,j,k))-=w(windex(i,j,k))*idz;
                        }
                    }
                    if (k<pindex.shape()(2)-1 && nz>0){       //direction z+
                        if (pindex(i,j,k+1)>-1 && windex(i,j,k+1)>-1){
                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/( ( wA.row(windex(i,j,k+1)).sum() )*dz2);
                            pA.insert(pindex(i,j,k),pindex(i,j,k+1)) = - 1/( ( wA.row(windex(i,j,k+1)).sum() )*dz2);
                            b(pindex(i,j,k))+=w(windex(i,j,k+1))*idz;
                        }
                    }

                    // and now the boundary conditions... pressure correction is zero at the inlet / outlet (imposed pressure)

                    if (i==0 && uindex(i,j,k)>-1){  //direction x-

                        pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.row(uindex(i,j,k)).sum() )*dx2);
                        b(pindex(i,j,k)) -= u(uindex(i,j,k))*idx;
                    }
                    if (i==pindex.shape()(0)-1 && uindex(i+1,j,k)>-1){  //direction x+

                        pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.row(uindex(i+1,j,k)).sum() )*dx2);
                        b(pindex(i,j,k)) += u(uindex(i+1,j,k))*idx;
                    }
                        // Nothing to add for the other directions, as the velocity is zero


                    //pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) /= alpha;

                }//end node found

            }//Inner for loop
        }
    }//outer for loop

    //Compression
    printf("OK\n");
    pA.makeCompressed();
    psolver.compute(pA);
    if(psolver.info()!=Eigen::Success) {
    // decomposition failed
        printf("Failure decomposing Matrix\n");

    }
}

void NavierStokesBase::AssemblePressureCorrectionSIMPLE(
                        Eigen::SparseMatrix<double,Eigen::RowMajor> & pA,
                        const Eigen::SparseMatrix<double,Eigen::RowMajor> & uA, const Eigen::SparseMatrix<double,Eigen::RowMajor> & vA, const Eigen::SparseMatrix<double,Eigen::RowMajor> & wA,
                        const Eigen::VectorXd &pvector, const Eigen::VectorXd &uc, const Eigen::VectorXd &vc, const Eigen::VectorXd &wc,
                        Eigen::VectorXd &b){

    /* Linear system associated with the continuity equation, mode= 0 */

    //~ std::cout<<"Assembling the linear system matrix for the pressure correction...";

    b.resize(pvector.innerSize());
    b = Eigen::VectorXd::Zero(pvector.innerSize());

    pA.resize(pvector.innerSize(),pvector.innerSize());
    pA.reserve(Eigen::VectorXi::Constant(pvector.innerSize(),7));

    double dx2,dy2,dz2,idx,idy,idz;

    int i,j,k;

    dx2=(dx*dx);
    dy2=(dy*dy);
    dz2=(dz*dz);
    idx=1/dx;
    idy=1/dy;
    idz=1/dz;

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k)

    // Traverse the image
    for (i=0;i<pindex.shape()(0);i++){
        for (j=0;j<pindex.shape()(1);j++){
            for (k=0;k<pindex.shape()(2);k++){

                if (pindex(i,j,k)>-1){   // Node found

                    pA.insert(pindex(i,j,k),pindex(i,j,k))=0;

                    // In the bulk
                    if(i>0){                        //direction x-
                        if (pindex(i-1,j,k)>-1 && uindex(i,j,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.coeff(uindex(i,j,k),uindex(i,j,k)) )*dx2);
                            pA.insert(pindex(i,j,k),pindex(i-1,j,k)) = - 1/(( uA.coeff(uindex(i,j,k),uindex(i,j,k)) )*dx2);
                            b(pindex(i,j,k))-=uc(uindex(i,j,k))*idx;
                        }
                    }
                    if (i<pindex.shape()(0)-1){       //direction x+
                        if (pindex(i+1,j,k)>-1 && uindex(i+1,j,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.coeff(uindex(i+1,j,k),uindex(i+1,j,k)) )*dx2);
                            pA.insert(pindex(i,j,k),pindex(i+1,j,k)) = - 1/(( uA.coeff(uindex(i+1,j,k),uindex(i+1,j,k)) )*dx2);
                            b(pindex(i,j,k))+=uc(uindex(i+1,j,k))*idx;
                        }
                    }
                    if(j>0){                        //direction y-
                        if (pindex(i,j-1,k)>-1 && vindex(i,j,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( vA.coeff(vindex(i,j,k),vindex(i,j,k)) )*dy2);
                            pA.insert(pindex(i,j,k),pindex(i,j-1,k)) = - 1/(( vA.coeff(vindex(i,j,k),vindex(i,j,k)) )*dy2);
                            b(pindex(i,j,k))-=vc(vindex(i,j,k))*idy;
                        }
                    }
                    if (j<pindex.shape()(1)-1){       //direction y+
                        if (pindex(i,j+1,k)>-1 && vindex(i,j+1,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( vA.coeff(vindex(i,j+1,k),vindex(i,j+1,k)) )*dy2);
                            pA.insert(pindex(i,j,k),pindex(i,j+1,k)) = - 1/(( vA.coeff(vindex(i,j+1,k),vindex(i,j+1,k)) )*dy2);
                            b(pindex(i,j,k))+=vc(vindex(i,j+1,k))*idy;
                        }
                    }
                    if(k>0 && nz>0){                        //direction z-
                        if (pindex(i,j,k-1)>-1 && windex(i,j,k)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( wA.coeff(windex(i,j,k),windex(i,j,k)) )*dz2);
                            pA.insert(pindex(i,j,k),pindex(i,j,k-1)) = - 1/(( wA.coeff(windex(i,j,k),windex(i,j,k)) )*dz2);
                            b(pindex(i,j,k))-=wc(windex(i,j,k))*idz;
                        }
                    }
                    if (k<pindex.shape()(2)-1 && nz>0){       //direction z+
                        if (pindex(i,j,k+1)>-1 && windex(i,j,k+1)>-1){

                            pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/( ( wA.coeff(windex(i,j,k+1),windex(i,j,k+1)) )*dz2);
                            pA.insert(pindex(i,j,k),pindex(i,j,k+1)) = - 1/( ( wA.coeff(windex(i,j,k+1),windex(i,j,k+1)) )*dz2);
                            b(pindex(i,j,k))+=wc(windex(i,j,k+1))*idz;
                        }
                    }

                    // and now the boundary conditions...

                    if (i==0 && uindex(i,j,k)>-1){  //direction x-
                        pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.coeff(uindex(i,j,k),uindex(i,j,k)) )*dx2);
                        b(pindex(i,j,k)) -= uc(uindex(i,j,k))*idx;
                    }
                    if (i==pindex.shape()(0)-1 && uindex(i+1,j,k)>-1){  //direction x+
                        pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) += 1/(( uA.coeff(uindex(i+1,j,k),uindex(i+1,j,k)) )*dx2);
                        b(pindex(i,j,k)) += uc(uindex(i+1,j,k))*idx;
                    }
                    // Nothing to add for the other directions, as the velocity is zero

                    //pA.coeffRef(pindex(i,j,k),pindex(i,j,k)) /= alpha;

                }//end node found

            }//Inner for loop
        }
    }//outer for loop

    pA.makeCompressed();

    psolver.compute(pA);
    if(psolver.info()!=Eigen::Success) {
    // decomposition failed
        printf("Failure decomposing Matrix\n");
    }

}

void NavierStokesBase::AssembleAllMomentum(
        Eigen::SparseMatrix<double,Eigen::RowMajor> & uA,
        Eigen::SparseMatrix<double,Eigen::RowMajor> & vA,
        Eigen::SparseMatrix<double,Eigen::RowMajor> & wA,
        Eigen::VectorXd &uc,
        Eigen::VectorXd &vc,
        Eigen::VectorXd &wc)
{

    AssembleMomentum_x(uA, uindex, uc, vc, wc);
    usolver.compute(uA);
    if(usolver.info()!=Eigen::Success) {
        printf("Failure decomposing Matrix uA\n");
    }//else std::cout<<"Matrix decomposition OK. "<<std::endl;

    AssembleMomentum_y(vA, vindex, uc, vc, wc);
    vsolver.compute(vA);
    if(vsolver.info()!=Eigen::Success) {
        printf("Failure decomposing Matrix vA\n");
    }

    if (nw>0){
        // w component
        AssembleMomentum_z(wA, windex, uc, vc, wc);
        wsolver.compute(wA);
        if(wsolver.info()!=Eigen::Success) {
            printf("Failure decomposing Matrix wA\n");
        }
    }


}

void NavierStokesBase::AssembleMomentum_x(Eigen::SparseMatrix<double,Eigen::RowMajor> & A, const blitz::Array<int,3>  & cindex, Eigen::VectorXd &uc, Eigen::VectorXd &vc, Eigen::VectorXd &wc){

    int i,j,k;
    double idx2,idy2,idz2,idx,idy,idz;
    double F,um,vm,wm,Fd;

    idx2=1/(dx*dx);
    idy2=1/(dy*dy);
    idz2=1/(dz*dz);
    idx=1/dx;
    idy=1/dy;
    idz=1/dz;

    A.resize(nu,nu);
    A.reserve(Eigen::VectorXi::Constant(nu,7));

    //~ std::cout<<"Assembling the matrix and the pseudo velocities u ";

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k,um,vm,wm,F,Fd)
    // Traverse the image - outside the external faces
    for (i=0;i<cindex.shape()(0);++i){
        for (j=0;j<cindex.shape()(1);++j){
            for (k=0;k<cindex.shape()(2);++k){
                if(cindex(i,j,k)>-1){               //Node found
                    A.insert(cindex(i,j,k),cindex(i,j,k))=0;

                    // Test the neighbors
                    if(i+1<cindex.shape()(0)){
                        if (cindex(i+1,j,k)>-1){

                            if(SafeVectorValue(uc,uindex,i,j,k)>0 && SafeVectorValue(uc,uindex,i+1,j,k)>0){
                                F = - SafeVectorValue(uc,uindex,i,j,k)*rho*idx;
                            }else if(SafeVectorValue(uc,uindex,i,j,k)<0 && SafeVectorValue(uc,uindex,i+1,j,k)<0){
                                F = - SafeVectorValue(uc,uindex,i+1,j,k)*rho*idx;
                            }else{
                                F=0;        // switch to a centered scheme if the two velocities are opposite
                            }

                            if (std::abs(F)*dx*dx > 2*mu*schemefactor){        // UPWIND SCHEME if Pe>2 on the east face (I,J,K)
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + std::min(F,0.0) ;
                                A.insert(cindex(i,j,k),cindex(i+1,j,k)) = mu*idx2 + std::max(F,0.0) ;
                            }else{
                                um = (SafeVectorValue(uc,uindex,i,j,k)+SafeVectorValue(uc,uindex,i+1,j,k))*0.5;
                                Fd= - um*rho*idx;
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + Fd*0.5 ;
                                A.insert(cindex(i,j,k),cindex(i+1,j,k)) = mu*idx2 + Fd*0.5 ;
                            }

                        }else{
                             A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 - (0.25*rho*idx)*(SafeVectorValue(uc,uindex,i,j,k));    // diagonal term, velocity zero at i+1
                        }
                    }else{ //outlet grad(uc)=0, u cst->u_n=u_n+1
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += - (rho*idx)*(SafeVectorValue(uc,uindex,i,j,k));
                    }

                    if(i-1>=0){
                        if (cindex(i-1,j,k)>-1){

                            if(SafeVectorValue(uc,uindex,i,j,k)<0 && SafeVectorValue(uc,uindex,i-1,j,k)<0){
                                F = SafeVectorValue(uc,uindex,i,j,k)*rho*idx;
                            }else if(SafeVectorValue(uc,uindex,i,j,k)>0 && SafeVectorValue(uc,uindex,i-1,j,k)>0){
                                F = SafeVectorValue(uc,uindex,i-1,j,k)*rho*idx;
                            }else{
                                F=0;
                            }

                            if (std::abs(F)*dx*dx > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + std::min(F,0.0);
                                A.insert(cindex(i,j,k),cindex(i-1,j,k)) = mu*idx2 + std::max(F,0.0);
                            }else{
                                um = (SafeVectorValue(uc,uindex,i,j,k)+SafeVectorValue(uc,uindex,i-1,j,k))*0.5;
                                Fd= um*rho*idx;
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + Fd*0.5;
                                A.insert(cindex(i,j,k),cindex(i-1,j,k)) = mu*idx2 + Fd*0.5;
                            }

                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + (0.25*rho*idx)*(SafeVectorValue(uc,uindex,i,j,k));
                        }
                    }else{  // boundary condition: grad(u)=0
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += (rho*idx)*(SafeVectorValue(uc,uindex,i,j,k));
                    }

                    if(j+1<cindex.shape()(1)){
                        if (cindex(i,j+1,k)>-1){
                            vm=(SafeVectorValue(vc,vindex,i-1,j+1,k)+SafeVectorValue(vc,vindex,i,j+1,k))*0.5;
                            F= - vm*rho*idy;

                            if (std::abs(vm)*rho*dy > 2*mu*schemefactor ){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += - mu*idy2 + std::min(F,0.0);    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i,j+1,k)) = mu*idy2 + std::max(F,0.0);    // extra-diagonal term
                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += - mu*idy2 + F*0.5;
                                A.insert(cindex(i,j,k),cindex(i,j+1,k)) = mu*idy2 + F*0.5;
                            }

                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idy2;        // velocity zero on the face
                        }
                    }else{
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idy2;        // face y+, velocity zero at i,j+1/2,k
                    }

                    if(j-1>=0){
                        if (cindex(i,j-1,k)>-1){
                            vm=(SafeVectorValue(vc,vindex,i-1,j,k)+SafeVectorValue(vc,vindex,i,j,k))*0.5;
                            F= vm*rho*idy;
                            if (std::abs(vm)*dy*rho > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + std::min(F,0.0);    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i,j-1,k)) = mu*idy2 + std::max(F,0.0); // extra-diagonal term
                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + F*0.5;    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i,j-1,k)) = mu*idy2 + F*0.5;    // extra-diagonal term
                            }

                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idy2;        // velocity zero on the face
                        }
                    }else{
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idy2;    // face y-, velocity zero at dy/2 from the unknown
                    }
                    if (pindex.shape()(2)>1){
                        if(k+1<cindex.shape()(2)){
                            if (cindex(i,j,k+1)>-1){
                                wm=(SafeVectorValue(wc,windex,i-1,j,k+1)+SafeVectorValue(wc,windex,i,j,k+1))*0.5;
                                F= - wm*rho*idz;
                                if (std::abs(wm)*rho*dz > 2*mu*schemefactor){
                                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + std::min(F,0.0);    // diagonal term
                                    A.insert(cindex(i,j,k),cindex(i,j,k+1)) = mu*idz2 + std::max(F,0.0);    // extra-diagonal term
                                }else{
                                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + F*0.5;    // diagonal term
                                    A.insert(cindex(i,j,k),cindex(i,j,k+1)) = mu*idz2 + F*0.5;    // extra-diagonal term
                                }

                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idz2;
                            }
                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idz2;
                        }

                        if(k-1>=0){
                            if (cindex(i,j,k-1)>-1){
                                wm=(SafeVectorValue(wc,windex,i-1,j,k)+SafeVectorValue(wc,windex,i,j,k))*0.5;
                                F= wm*rho*idz;
                                if (std::abs(wm)*rho*dz > 2*mu*schemefactor){
                                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + std::min(F,0.0);    // diagonal term
                                    A.insert(cindex(i,j,k),cindex(i,j,k-1)) = mu*idz2 + std::max(F,0.0);    // extra-diagonal term
                                }else{
                                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + F*0.5;    // diagonal term
                                    A.insert(cindex(i,j,k),cindex(i,j,k-1)) = mu*idz2 + F*0.5;    // extra-diagonal term
                                }   // extra-diagonal term

                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idz2;
                            }
                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idz2;
                        }
                    }
                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) /= alpha;

                }// Test index!=-1

            }// inner z loop
        }// y loop
    }// outer x loop

    A.makeCompressed();

}

void NavierStokesBase::AssembleMomentum_y(Eigen::SparseMatrix<double,Eigen::RowMajor> & A, const blitz::Array<int,3>  & cindex, Eigen::VectorXd &uc, Eigen::VectorXd &vc, Eigen::VectorXd &wc){

    int i,j,k;
    double idx2,idy2,idz2,idx,idy,idz;
    double F,um,vm,wm,Fd;

    idx2=1/(dx*dx);
    idy2=1/(dy*dy);
    idz2=1/(dz*dz);
    idx=1/dx;
    idy=1/dy;
    idz=1/dz;

    A.resize(nv,nv);
    A.reserve(Eigen::VectorXi::Constant(nv,7));

    //~ std::cout<<"Assembling the matrix and the pseudo velocities v ";

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k,um,vm,wm,F,Fd)
    // Traverse the image - outside the external faces
    for (i=0;i<cindex.shape()(0);i++){
        for (j=0;j<cindex.shape()(1);j++){
            for (k=0;k<cindex.shape()(2);++k){
                if(cindex(i,j,k)>-1){               //Node found
                    A.insert(cindex(i,j,k),cindex(i,j,k))=0;
                    // Test the neighbors

                    if(i+1<cindex.shape()(0)){
                        if (cindex(i+1,j,k)>-1){
                            um =(SafeVectorValue(uc,uindex,i+1,j-1,k)+SafeVectorValue(uc,uindex,i+1,j,k))*0.5;
                            F = - um*rho*idx;
                            if (std::abs(um)*dx*rho > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + std::min(F,0.0);    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i+1,j,k)) = mu*idx2 + std::max(F,0.0) ;    // extra-diagonal term
                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + F*0.5;    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i+1,j,k)) = mu*idx2 + F*0.5 ;
                            }


                        }else{
                             A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idx2;    // unknown v_I,j,K, interface at dx/2 (i,j,K), velocity zero
                        }
                    }else{ //outlet
                        //~ A.coeffRef(cindex(i,j,k),cindex(i,j,k)) +=  //-mu*idx2 -(0.25*rho*idx)*(SafeVectorValue(uc,uindex,i+1,j-1,k)+SafeVectorValue(uc,uindex,i+1,j,k)); velocity at nx+1 zero
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -(0.5*rho*idx)*(SafeVectorValue(uc,uindex,i+1,j-1,k)+SafeVectorValue(uc,uindex,i+1,j,k)); // zero gradient
                    }

                    if(i-1>=0){
                        if (cindex(i-1,j,k)>-1){
                            um = (SafeVectorValue(uc,uindex,i,j-1,k)+SafeVectorValue(uc,uindex,i,j,k))*0.5;
                            F = um*rho*idx;
                            if (std::abs(um)*dx*rho > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + std::min(F,0.0);    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i-1,j,k)) = mu*idx2 + std::max(F,0.0) ;    // extra-diagonal term
                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + F*0.5;    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i-1,j,k)) = mu*idx2 + F*0.5 ;
                            }


                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idx2;
                        }
                    }else{  //inlet
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) +=  -mu*idx2 +(0.25*rho*idx)*(SafeVectorValue(uc,uindex,i,j-1,k)+SafeVectorValue(uc,uindex,i,j,k)); // velocity zero at 0
                    }
                    if(j+1<cindex.shape()(1)){
                        if (cindex(i,j+1,k)>-1){

                            if(SafeVectorValue(vc,vindex,i,j,k)>0 && SafeVectorValue(vc,vindex,i,j+1,k)>0){
                                F = - SafeVectorValue(vc,vindex,i,j,k)*rho*idy;
                            }else if(SafeVectorValue(vc,vindex,i,j,k)<0 && SafeVectorValue(vc,vindex,i,j+1,k)<0){
                                F = - SafeVectorValue(vc,vindex,i,j+1,k)*rho*idy;
                            }else{
                                F=0;
                            }

                            if (std::abs(F)*dy*dy > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + std::min(F,0.0);
                                A.insert(cindex(i,j,k),cindex(i,j+1,k)) = mu*idy2 + std::max(F,0.0);
                            }else{
                                vm = (SafeVectorValue(vc,vindex,i,j+1,k)+SafeVectorValue(vc,vindex,i,j,k))*0.5;
                                Fd = - vm*rho*idy;
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + Fd*0.5;
                                A.insert(cindex(i,j,k),cindex(i,j+1,k)) = mu*idy2 + Fd*0.5;
                            }


                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 - (0.25*rho*idy)*(SafeVectorValue(vc,vindex,i,j,k));        // velocity zero on the face, interface at dy
                        }
                    }else{
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 - (0.25*rho*idy)*(SafeVectorValue(vc,vindex,i,j,k));
                    }

                    if(j-1>=0){

                        if (cindex(i,j-1,k)>-1){

                            if(SafeVectorValue(vc,vindex,i,j,k)<0 && SafeVectorValue(vc,vindex,i,j-1,k)<0){
                                F = SafeVectorValue(vc,vindex,i,j,k)*rho*idy;
                            }else if(SafeVectorValue(vc,vindex,i,j,k)>0 && SafeVectorValue(vc,vindex,i,j-1,k)>0){
                                F = SafeVectorValue(vc,vindex,i,j-1,k)*rho*idy;
                            }else{
                                F=0;
                            }
                            if (std::abs(F)*dy*dy > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + std::min(F,0.0);
                                A.insert(cindex(i,j,k),cindex(i,j-1,k)) = mu*idy2 + std::max(F,0.0);
                            }else{
                                vm = (SafeVectorValue(vc,vindex,i,j-1,k)+SafeVectorValue(vc,vindex,i,j,k))*0.5;
                                Fd = vm*rho*idy;
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + Fd*0.5;
                                A.insert(cindex(i,j,k),cindex(i,j-1,k)) = mu*idy2 + Fd*0.5;
                            }


                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + (0.25*rho*idy)*(SafeVectorValue(vc,vindex,i,j,k));
                        }
                    }else{
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + (0.25*rho*idy)*(SafeVectorValue(vc,vindex,i,j,k));
                    }

                    if (pindex.shape()(2)>1){
                        if(k+1<cindex.shape()(2)){
                            if (cindex(i,j,k+1)>-1){
                                wm = (SafeVectorValue(wc,windex,i,j-1,k+1)+SafeVectorValue(wc,windex,i,j,k+1))*0.5;
                                F = - wm*rho*idz;
                                if (std::abs(wm)*rho*dz > 2*mu*schemefactor){
                                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + std::min(F,0.0);
                                    A.insert(cindex(i,j,k),cindex(i,j,k+1)) = mu*idz2 + std::max(F,0.0);
                                }else{
                                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + F*0.5;
                                    A.insert(cindex(i,j,k),cindex(i,j,k+1)) = mu*idz2 + F*0.5;
                                }


                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idz2;
                            }
                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idz2;
                        }

                        if(k-1>=0){
                            if (cindex(i,j,k-1)>-1){
                                wm = (SafeVectorValue(wc,windex,i,j-1,k)+SafeVectorValue(wc,windex,i,j,k))*0.5;
                                F = wm*rho*idz;
                                if (std::abs(wm)*rho*dz > 2*mu*schemefactor){
                                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + std::min(F,0.0);
                                    A.insert(cindex(i,j,k),cindex(i,j,k-1)) = mu*idz2 + std::max(F,0.0);
                                }else{
                                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + F*0.5;
                                    A.insert(cindex(i,j,k),cindex(i,j,k-1)) = mu*idz2 + F*0.5;
                                }


                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idz2;
                            }
                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idz2;
                        }
                    }

                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) /= alpha;

                }// Test index!=-1

            }// inner z loop
        }// y loop
    }// outer x loop

    //~ #pragma omp parallel for default(shared) private(i,j,k)
    //~ for (i=0;i<cindex.shape()(0);i++){
        //~ for (j=0;j<cindex.shape()(1);j++){
            //~ for (k=0;k<cindex.shape()(2);++k){
                //~ if(cindex(i,j,k)>-1){               //Node found
                    //~ A.coeffRef(cindex(i,j,k),cindex(i,j,k)) /= alpha;
                //~ }
            //~ }
        //~ }
    //~ }

    A.makeCompressed();


    //~ std::cout<<"...OK"<<std::endl;
}

void NavierStokesBase::AssembleMomentum_z(Eigen::SparseMatrix<double,Eigen::RowMajor> & A, const blitz::Array<int,3>  & cindex, Eigen::VectorXd &uc, Eigen::VectorXd &vc, Eigen::VectorXd &wc){


    int i,j,k;
    double idx2,idy2,idz2,idx,idy,idz;
    double F,um,vm,wm,Fd;

    idx2=1/(dx*dx);
    idy2=1/(dy*dy);
    idz2=1/(dz*dz);
    idx=1/dx;
    idy=1/dy;
    idz=1/dz;

    A.resize(nw,nw);
    A.reserve(Eigen::VectorXi::Constant(nw,7));

    //~ std::cout<<"Assembling the matrix and the pseudo velocities w ";

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k,um,vm,wm,F,Fd)
    // Traverse the image - outside the external faces
    for (i=0;i<cindex.shape()(0);i++){
        for (j=0;j<cindex.shape()(1);j++){
            for (k=0;k<cindex.shape()(2);++k){
                if(cindex(i,j,k)>-1){               //Node found
                    A.insert(cindex(i,j,k),cindex(i,j,k))=0;
                    // Test the neighbors

                    if(i+1<cindex.shape()(0)){
                        if (cindex(i+1,j,k)>-1){
                            um =(SafeVectorValue(uc,uindex,i+1,j,k-1)+SafeVectorValue(uc,uindex,i+1,j,k))*0.5;
                            F = - um*rho*idx;
                            if (std::abs(um)*dx*rho > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + std::min(F,0.0);    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i+1,j,k)) = mu*idx2 + std::max(F,0.0) ;    // extra-diagonal term
                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + F*0.5;    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i+1,j,k)) = mu*idx2 + F*0.5 ;
                            }


                        }else{
                             A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idx2;    // unknown v_I,j,K, interface at dx/2 (i,j,K), velocity zero
                        }
                    }else{ //outlet grad(uc)=0
                        //~ A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 -(0.25*rho*idx)*(SafeVectorValue(uc,uindex,i+1,j,k-1)+SafeVectorValue(uc,uindex,i+1,j,k)); // component zero at nx+1
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -(0.5*rho*idx)*(SafeVectorValue(uc,uindex,i+1,j,k-1)+SafeVectorValue(uc,uindex,i+1,j,k)); // zero gradient
                    }

                    if(i-1>=0){
                        if (cindex(i-1,j,k)>-1){
                            um = (SafeVectorValue(uc,uindex,i,j,k-1)+SafeVectorValue(uc,uindex,i,j,k))*0.5;
                            F = um*rho*idx;
                            if (std::abs(um)*dx*rho > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + std::min(F,0.0);    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i-1,j,k)) = mu*idx2 + std::max(F,0.0) ;    // extra-diagonal term
                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + F*0.5;    // diagonal term
                                A.insert(cindex(i,j,k),cindex(i-1,j,k)) = mu*idx2 + F*0.5 ;
                            }


                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idx2;
                        }
                    }else{  // boundary condition: grad(uc)=0
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idx2 + (0.25*rho*idx)*(SafeVectorValue(uc,uindex,i,j,k-1)+SafeVectorValue(uc,uindex,i,j,k)); // component zero at 0
                    }

                    if(j+1<cindex.shape()(1)){
                        if (cindex(i,j+1,k)>-1){
                            vm = (SafeVectorValue(vc,vindex,i,j+1,k-1)+SafeVectorValue(vc,vindex,i,j+1,k))*0.5;
                            F = - vm*rho*idy;
                            if (std::abs(vm)*rho*dy > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + std::min(F,0.0);
                                A.insert(cindex(i,j,k),cindex(i,j+1,k)) = mu*idy2 + std::max(F,0.0);
                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + F*0.5;
                                A.insert(cindex(i,j,k),cindex(i,j+1,k)) = mu*idy2 + F*0.5;
                            }


                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idy2;        // velocity zero on the face
                        }
                    }else{
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idy2;        // face y+, velocity zero at dz/2 from the unknown
                    }

                    if(j-1>=0){
                        if (cindex(i,j-1,k)>-1){
                            vm = (SafeVectorValue(vc,vindex,i,j,k-1)+SafeVectorValue(vc,vindex,i,j,k))*0.5;
                            F = vm*rho*idy;
                            if (std::abs(vm)*rho*dy > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + std::min(F,0.0);
                                A.insert(cindex(i,j,k),cindex(i,j-1,k)) = mu*idy2 + std::max(F,0.0);
                            }else{
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idy2 + F*0.5;
                                A.insert(cindex(i,j,k),cindex(i,j-1,k)) = mu*idy2 + F*0.5;
                            }


                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idy2;        // velocity zero on the face
                        }
                    }else{
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -2*mu*idy2;    // face y-, velocity zero at dz/2 from the unknown
                    }

                    if(k+1<cindex.shape()(2)){
                        if (cindex(i,j,k+1)>-1){

                            if(SafeVectorValue(wc,windex,i,j,k)>0 && SafeVectorValue(wc,windex,i,j,k+1)>0){
                                F = - SafeVectorValue(wc,windex,i,j,k)*rho*idz;
                            }else if(SafeVectorValue(wc,windex,i,j,k)<0 && SafeVectorValue(wc,windex,i,j,k+1)<0){
                                F = - SafeVectorValue(wc,windex,i,j,k+1)*rho*idz;
                            }else{
                                F=0;
                            }

                            if (std::abs(F)*dz*dz > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + std::min(F,0.0);
                                A.insert(cindex(i,j,k),cindex(i,j,k+1)) = mu*idz2 + std::max(F,0.0);
                            }else{
                                wm = (SafeVectorValue(wc,windex,i,j,k)+SafeVectorValue(wc,windex,i,j,k+1))*0.5;
                                Fd = - wm*rho*idz;
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + Fd*0.5;
                                A.insert(cindex(i,j,k),cindex(i,j,k+1)) = mu*idz2 + Fd*0.5;
                            }


                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 - (0.25*rho*idz)*(SafeVectorValue(wc,windex,i,j,k));
                        }
                    }else{
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 - (0.25*rho*idz)*(SafeVectorValue(wc,windex,i,j,k));
                    }

                    if(k-1>=0){
                        if (cindex(i,j,k-1)>-1){

                            if(SafeVectorValue(wc,windex,i,j,k-1)<0 && SafeVectorValue(wc,windex,i,j,k)<0){
                                F =  SafeVectorValue(wc,windex,i,j,k)*rho*idz;
                            }else if(SafeVectorValue(wc,windex,i,j,k-1)>0 && SafeVectorValue(wc,windex,i,j,k)>0){
                                F =  SafeVectorValue(wc,windex,i,j,k-1)*rho*idz;
                            }else{
                                F = 0;
                            }
                            if (std::abs(F)*dz*dz > 2*mu*schemefactor){
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + std::min(F,0.0);
                                A.insert(cindex(i,j,k),cindex(i,j,k-1)) = mu*idz2 + std::max(F,0.0);
                            }else{
                                wm = (SafeVectorValue(wc,windex,i,j,k-1)+SafeVectorValue(wc,windex,i,j,k))*0.5;
                                Fd = wm*rho*idz;
                                A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + Fd*0.5;
                                A.insert(cindex(i,j,k),cindex(i,j,k-1)) = mu*idz2 + Fd*0.5;
                            }


                        }else{
                            A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + (0.25*rho*idz)*(SafeVectorValue(wc,windex,i,j,k));
                        }
                    }else{
                        A.coeffRef(cindex(i,j,k),cindex(i,j,k)) += -mu*idz2 + (0.25*rho*idz)*(SafeVectorValue(wc,windex,i,j,k));
                    }
                    A.coeffRef(cindex(i,j,k),cindex(i,j,k)) /= alpha;


                }// Test index!=-1

            }// inner z loop
        }// y loop
    }// outer x loop

    //~ #pragma omp parallel for default(shared) private(i,j,k)
    //~ for (i=0;i<cindex.shape()(0);i++){
        //~ for (j=0;j<cindex.shape()(1);j++){
            //~ for (k=0;k<cindex.shape()(2);++k){
                //~ if(cindex(i,j,k)>-1){               //Node found
                    //~ A.coeffRef(cindex(i,j,k),cindex(i,j,k)) /= alpha;
                //~ }
            //~ }
        //~ }
    //~ }

    A.makeCompressed();

    //~ std::cout<<"...OK"<<std::endl;

}


void NavierStokesBase::Assemble_RHS_u(
            const Eigen::SparseMatrix<double,Eigen::RowMajor> & uA,
            const Eigen::VectorXd &pvector,
            const Eigen::VectorXd &u,
            Eigen::VectorXd &b){

    int i,j,k;
    double idx;

    idx=1/dx;

    b.resize(u.innerSize());
    b = Eigen::VectorXd::Zero(nu);

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k)
    // Traverse the image - outside the external faces
    for (i=0;i<uindex.shape()(0);++i){
        for (j=0;j<uindex.shape()(1);++j){
            for (k=0;k<uindex.shape()(2);++k){
                if(uindex(i,j,k)>-1){               //Node found
                    // relaxation of the diagonal coefficients - already done
                    //uA.coeffRef(uindex(i,j,k),uindex(i,j,k)) /= alpha;

                    if(i==0) {
                        b(uindex(i,j,k)) = (pvector(pindex(i,j,k))-p0)*idx;
                    }else if (i==pindex.shape()(0)){
                        b(uindex(i,j,k)) = (p1-pvector(pindex(i-1,j,k)))*idx;
                    }else if (i<pindex.shape()(0) && i>0){
                        b(uindex(i,j,k)) = (pvector(pindex(i,j,k))-pvector(pindex(i-1,j,k)))*idx;
                        //~ std::cout<<i<<","<<j<<","<<k<<","<<pindex(i,j,k)<<","<< pindex(i-1,j,k) <<","<<pvector(pindex(i,j,k)) <<", "<<pvector(pindex(i-1,j,k)) <<std::endl;
                    }else{
                        printf("Problem !\n");
                    }
                    // relaxation of the right-hand side
                    b(uindex(i,j,k)) += u(uindex(i,j,k))*uA.coeff(uindex(i,j,k),uindex(i,j,k))*(1-alpha);
                }
            }
        }
    }


}

void NavierStokesBase::Assemble_RHS_v(const Eigen::SparseMatrix<double,Eigen::RowMajor> & vA,
                    const Eigen::VectorXd &pvector, Eigen::VectorXd &v,Eigen::VectorXd &b){

    int i,j,k;
    double idy;

    idy=1/dy;

    b.resize(v.innerSize());
    b = Eigen::VectorXd::Zero(nv);

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k)
    // Traverse the image
    for (i=0;i<vindex.shape()(0);i++){
        for (j=0;j<vindex.shape()(1);j++){
            for (k=0;k<vindex.shape()(2);++k){
                if(vindex(i,j,k)>-1){               //Node found

                    // relaxation of the diagonal coefficients
                    //vA.coeffRef(vindex(i,j,k),vindex(i,j,k)) /= alpha;

                    b(vindex(i,j,k)) = (pvector(pindex(i,j,k))-pvector(pindex(i,j-1,k)))*idy + v(vindex(i,j,k))*vA.coeff(vindex(i,j,k),vindex(i,j,k))*(1-alpha);  // always OK because vindex(i,0,k)==-1

                }
            }
        }
    }

}

void NavierStokesBase::Assemble_RHS_w(const Eigen::SparseMatrix<double,Eigen::RowMajor> & wA,
                    const Eigen::VectorXd &pvector, Eigen::VectorXd &w, Eigen::VectorXd &b){

    int i,j,k;
    double idz;

    idz=1/dz;

    b.resize(w.innerSize());
    b = Eigen::VectorXd::Zero(nw);

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k)
    // Traverse the image
    for (i=0;i<windex.shape()(0);i++){
        for (j=0;j<windex.shape()(1);j++){
            for (k=0;k<windex.shape()(2);++k){
                if(windex(i,j,k)>-1){               //Node found
                    // relaxation of the diagonal coefficients
                    //wA.coeffRef(windex(i,j,k),windex(i,j,k)) /= alpha;
                    b(windex(i,j,k)) = (pvector(pindex(i,j,k))-pvector(pindex(i,j,k-1)))*idz + w(windex(i,j,k))*wA.coeff(windex(i,j,k),windex(i,j,k))*(1-alpha);  // always OK because windex(i,j,0)==-1
                }
            }
        }
    }

}

void NavierStokesBase::Compute_u_hat(
                    Eigen::VectorXd &u_hat,
                    const Eigen::SparseMatrix<double,Eigen::RowMajor> &A,
                    const Eigen::VectorXd &uc,
                    const Eigen::VectorXd &u_old){

    int i,j,k;

    //~ std::cout<<"Assembling the matrix and the pseudo velocities u ";

    u_hat = Eigen::VectorXd::Zero(nu);

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k)
    // Traverse the image - outside the external faces
    for (i=0;i<uindex.shape()(0);++i){
        for (j=0;j<uindex.shape()(1);++j){
            for (k=0;k<uindex.shape()(2);++k){
                if(uindex(i,j,k)>-1){               //Node found


                    // Test the neighbors
                    if(i+1<uindex.shape()(0)){
                        if (uindex(i+1,j,k)>-1){
                            u_hat(uindex(i,j,k)) -= A.coeff(uindex(i,j,k),uindex(i+1,j,k))*uc(uindex(i+1,j,k));
                        }
                    }

                    if(i-1>=0){
                        if (uindex(i-1,j,k)>-1){
                            u_hat(uindex(i,j,k)) -= A.coeff(uindex(i,j,k),uindex(i-1,j,k))*uc(uindex(i-1,j,k));
                        }
                    }

                    if(j+1<uindex.shape()(1)){
                        if (uindex(i,j+1,k)>-1){
                            u_hat(uindex(i,j,k)) -= A.coeff(uindex(i,j,k),uindex(i,j+1,k))*uc(uindex(i,j+1,k));
                        }
                    }

                    if(j-1>=0){
                        if (uindex(i,j-1,k)>-1){
                            u_hat(uindex(i,j,k)) -= A.coeff(uindex(i,j,k),uindex(i,j-1,k))*uc(uindex(i,j-1,k));
                        }
                    }
                    if (pindex.shape()(2)>1){
                        if(k+1<uindex.shape()(2)){
                            if (uindex(i,j,k+1)>-1){
                                u_hat(uindex(i,j,k)) -= A.coeff(uindex(i,j,k),uindex(i,j,k+1))*uc(uindex(i,j,k+1));
                            }
                        }

                        if(k-1>=0){
                            if (uindex(i,j,k-1)>-1){
                                u_hat(uindex(i,j,k)) -= A.coeff(uindex(i,j,k),uindex(i,j,k-1))*uc(uindex(i,j,k-1));
                            }
                        }
                    }


                    // u_n-1, A(i,i) is already relaxed
                    u_hat(uindex(i,j,k)) = (u_hat(uindex(i,j,k)))/(A.coeff(uindex(i,j,k),uindex(i,j,k))) + u_old(uindex(i,j,k))*(1-alpha);

                }// Test index!=-1

            }// inner z loop
        }// y loop
    }// outer x loop

}


void NavierStokesBase::Compute_v_hat(
                    Eigen::VectorXd &v_hat,
                    const Eigen::SparseMatrix<double,Eigen::RowMajor> &A,
                    const Eigen::VectorXd &vc,
                    const Eigen::VectorXd &v_old){

    int i,j,k;

    v_hat = Eigen::VectorXd::Zero(nv);

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k)
    // Traverse the image - outside the external faces
    for (i=0;i<vindex.shape()(0);i++){
        for (j=0;j<vindex.shape()(1);j++){
            for (k=0;k<vindex.shape()(2);++k){
                if(vindex(i,j,k)>-1){               //Node found
                    // Test the neighbors

                    if(i+1<vindex.shape()(0)){
                        if (vindex(i+1,j,k)>-1){
                            v_hat(vindex(i,j,k)) -= A.coeff(vindex(i,j,k),vindex(i+1,j,k))*vc(vindex(i+1,j,k));
                        }
                    }

                    if(i-1>=0){
                        if (vindex(i-1,j,k)>-1){
                            v_hat(vindex(i,j,k)) -= A.coeff(vindex(i,j,k),vindex(i-1,j,k))*vc(vindex(i-1,j,k));
                        }
                    }
                    if(j+1<vindex.shape()(1)){
                        if (vindex(i,j+1,k)>-1){
                            v_hat(vindex(i,j,k)) -= A.coeff(vindex(i,j,k),vindex(i,j+1,k))*vc(vindex(i,j+1,k));
                        }
                    }

                    if(j-1>=0){

                        if (vindex(i,j-1,k)>-1){
                            v_hat(vindex(i,j,k)) -= A.coeff(vindex(i,j,k),vindex(i,j-1,k))*vc(vindex(i,j-1,k));
                        }
                    }

                    if (pindex.shape()(2)>1){
                        if(k+1<vindex.shape()(2)){
                            if (vindex(i,j,k+1)>-1){
                                v_hat(vindex(i,j,k)) -= A.coeff(vindex(i,j,k),vindex(i,j,k+1))*vc(vindex(i,j,k+1));

                            }
                        }

                        if(k-1>=0){
                            if (vindex(i,j,k-1)>-1){

                                v_hat(vindex(i,j,k)) -= A.coeff(vindex(i,j,k),vindex(i,j,k-1))*vc(vindex(i,j,k-1));

                            }
                        }
                    }

                    //~ A.coeffRef(vindex(i,j,k),vindex(i,j,k)) /= alpha;
                    v_hat(vindex(i,j,k)) = v_hat(vindex(i,j,k))/(A.coeff(vindex(i,j,k),vindex(i,j,k)))+v_old(vindex(i,j,k))*(1-alpha);

                }// Test index!=-1

            }// inner z loop
        }// y loop
    }// outer x loop

}

void NavierStokesBase::Compute_w_hat(
                    Eigen::VectorXd &w_hat,
                    const Eigen::SparseMatrix<double,Eigen::RowMajor> &A,
                    const Eigen::VectorXd &wc,
                    const Eigen::VectorXd &w_old){

    int i,j,k;

    w_hat = Eigen::VectorXd::Zero(nw);

    //~ Assembling matrix
    #pragma omp parallel for default(shared) private(i,j,k)
    // Traverse the image - outside the external faces
    for (i=0;i<windex.shape()(0);i++){
        for (j=0;j<windex.shape()(1);j++){
            for (k=0;k<windex.shape()(2);++k){
                if(windex(i,j,k)>-1){               //Node found
                    // Test the neighbors

                    if(i+1<windex.shape()(0)){
                        if (windex(i+1,j,k)>-1){
                            w_hat(windex(i,j,k))-=A.coeff(windex(i,j,k),windex(i+1,j,k))*wc(windex(i+1,j,k));
                        }
                    }

                    if(i-1>=0){
                        if (windex(i-1,j,k)>-1){
                            w_hat(windex(i,j,k))-=A.coeff(windex(i,j,k),windex(i-1,j,k))*wc(windex(i-1,j,k));
                        }
                    }

                    if(j+1<windex.shape()(1)){
                        if (windex(i,j+1,k)>-1){
                            w_hat(windex(i,j,k))-=A.coeff(windex(i,j,k),windex(i,j+1,k))*wc(windex(i,j+1,k));
                        }
                    }

                    if(j-1>=0){
                        if (windex(i,j-1,k)>-1){
                            w_hat(windex(i,j,k))-=A.coeff(windex(i,j,k),windex(i,j-1,k))*wc(windex(i,j-1,k));
                        }
                    }

                    if(k+1<windex.shape()(2)){
                        if (windex(i,j,k+1)>-1){
                            w_hat(windex(i,j,k))-=A.coeff(windex(i,j,k),windex(i,j,k+1))*wc(windex(i,j,k+1));
                        }
                    }

                    if(k-1>=0){
                        if (windex(i,j,k-1)>-1){
                            w_hat(windex(i,j,k))-=A.coeff(windex(i,j,k),windex(i,j,k-1))*wc(windex(i,j,k-1));
                        }
                    }
                    //~ A.coeffRef(windex(i,j,k),windex(i,j,k)) /= alpha;
                    w_hat(windex(i,j,k))= w_hat(windex(i,j,k))/(A.coeff(windex(i,j,k),windex(i,j,k))) + w_old(windex(i,j,k))*(1-alpha);


                }// Test index!=-1

            }// inner z loop
        }// y loop
    }// outer x loop

}

double NavierStokesBase::SafeVectorValue(const Eigen::VectorXd & vec, const blitz::Array<int,3>  &indexvector, int i, int j, int k){

    //~ Return the value of the vector at given index if it exists. Return 0 otherwise.

    if (i>=indexvector.shape()(0) || i <0 || j>=indexvector.shape()(1) || j <0 || k>=indexvector.shape()(2) || k <0){
        return 0;
    }
    else{
        if (indexvector(i,j,k)>-1){
            return vec(indexvector(i,j,k)) ;
        }else{
            return 0;
        }
    }
}

void NavierStokesBase::ComputeVelocityField(
                Eigen::SparseMatrix<double,Eigen::RowMajor> & uA,
                Eigen::SparseMatrix<double,Eigen::RowMajor> & vA,
                Eigen::SparseMatrix<double,Eigen::RowMajor> & wA,
                Eigen::VectorXd &u,
                Eigen::VectorXd &v,
                Eigen::VectorXd &w,
                const Eigen::VectorXd &pvector,
                bool limit,
                double delta_v_max){

    bool retry;

    Eigen::VectorXd temp;
    Eigen::VectorXd b;

    double delta ;
    int i;
    retry=true;

    while (retry){

        retry = false;
        Assemble_RHS_u(uA, pvector, u, b);
        temp.resize(nu);
        temp = u;
        u = usolver.solveWithGuess(b,u);
        if (usolver.iterations() == 0 || usolver.info()!=0){
            u = temp;
        }

        if (limit == true){
            #pragma omp parallel for default(shared) private(i,delta)
            for (i = 0; i<u.size(); i++){
                delta = u(i)-temp(i);
                if (std::abs(delta) > delta_v_max){
                    //std::cout<<"limiting velocity update delta_v="<<delta<<std::endl;
                    if (delta > 0){
                        u(i) = temp(i) + delta_v_max;
                    }
                    else if (delta <0){
                        u(i) = temp(i) - delta_v_max;
                    }
                }
            }
        }
        printf("Solving for component u in %d iterations (status %d)\n", (int)usolver.iterations(), (int)usolver.info());
        info_file<<"Solving for component u in "<<usolver.iterations()<<" iterations"<<" (status "<< usolver.info()<< ")"<<std::endl;

        Assemble_RHS_v(vA, pvector, v, b);
        temp.resize(nv);
        temp = v;
        v=vsolver.solveWithGuess(b,v);
        if (vsolver.iterations() == 0 || vsolver.info()!=0){
            v = temp;
        }
        if (limit == true){
            #pragma omp parallel for default(shared) private(i,delta)
            for (i = 0; i<v.size(); i++){
                delta = v(i)-temp(i);
                if (std::abs(delta) > delta_v_max){
                    //std::cout<<"limiting velocity update delta_v="<<delta<<std::endl;
                    if (delta > 0){
                        v(i) = temp(i) + delta_v_max;
                    }
                    else if (delta <0){
                        v(i) = temp(i) - delta_v_max;
                    }
                }
            }
        }

        printf("Solving for component v in %d iterations (status %d)\n", (int)vsolver.iterations(), (int)vsolver.info());
        info_file<<"Solving for component v in "<<vsolver.iterations()<<" iterations"<<" (status "<< vsolver.info()<< ")"<<std::endl;

        if(nw>0){
            Assemble_RHS_w(wA, pvector, w, b);
            temp.resize(nw);
            temp = w;
            w=wsolver.solveWithGuess(b,w);
            if (wsolver.iterations() == 0 || wsolver.info()!=0){
                w = temp;
            }

            if (limit == true){
            #pragma omp parallel for default(shared) private(i,delta)
            for (i = 0; i<w.size(); i++){
                delta = w(i)-temp(i);
                if (std::abs(delta) > delta_v_max){
                    //std::cout<<"limiting velocity update delta_v="<<delta<<std::endl;
                    if (delta > 0){
                        w(i) = temp(i) + delta_v_max;
                    }
                    else if (delta <0){
                        w(i) = temp(i) - delta_v_max;
                    }
                }
            }
        }

            printf("Solving for component w in %d iterations (status %d)\n", (int)wsolver.iterations(), (int)wsolver.info());
            info_file<<"Solving for component w in "<<wsolver.iterations()<<" iterations"<<" (status "<< wsolver.info()<< ")"<<std::endl;
        }

        //if (usolver.info() !=0 || vsolver.info() !=0 || wsolver.info() !=0){
            //alpha*=0.9;
            //std::cout<<"Relaxing alpha :"<<alpha<<std::endl;
            //retry=true;
            //AssembleAllMomentum(uA,vA,wA,u,v,w);
        //}
    }
}

void NavierStokesBase::ComputePermeability(double &Q, double &v_moy, double &K, const Eigen::VectorXd &u, const Eigen::VectorXd & pvector){

    //Compute the permeability

    double sx, p;
    int np_in=0,np_out=0;
    int j,k;

    sx=dy*dz;
    p=0;
    v_moy=0;



    if (border <= 0){
        for (j=0;j<uindex.shape()(1);++j){
            for (k=0;k<uindex.shape()(2);++k){
                if (uindex(uindex.shape()(0)-1,j,k)>-1){
                    v_moy+=u(uindex(uindex.shape()(0)-1,j,k));
                    p+=1;
                }
            }
        }

        p_inlet = p0;
        p_outlet = p1;

    }else{

        p_inlet=0;
        p_outlet=0;

        for (j=0;j<uindex.shape()(1);++j){
            for (k=0;k<uindex.shape()(2);++k){
                if (uindex(uindex.shape()(0)-border,j,k)>-1){
                    v_moy+=u(uindex(uindex.shape()(0)-border,j,k));
                    p+=1;
                }
            }
        }

        for (j=0;j<pindex.shape()(1);++j){
            for (k=0;k<pindex.shape()(2);++k){
                if (pindex(pindex.shape()(0)-border,j,k)>-1){
                    p_outlet += pvector(pindex(pindex.shape()(0)-border,j,k));
                    np_out += 1;
                }
                if(pindex(border-1,j,k)>-1){
                    p_inlet += pvector(pindex(border-1,j,k));
                    np_in += 1;
                }
            }
        }

        if (np_in > 0) p_inlet /= np_in;
        if (np_out > 0) p_outlet /= np_out;

    }

    //~ p=0.5*p/(uindex.shape()(1)*uindex.shape()(2));
    //~ Q*=0.5;
    Q=v_moy*sx;
    p/=(uindex.shape()(1)*uindex.shape()(2));
    v_moy/=(uindex.shape()(1)*uindex.shape()(2));
    if (std::abs(p_inlet-p_outlet) > 0.0){
        K=v_moy*mu*dx*(pindex.shape()(0)-2*border)/(p_inlet-p_outlet);
    }else{
        K=0.0;
        printf("WARNING: zero pressure difference (p_inlet = p_outlet), K set to 0\n");
    }
}

void NavierStokesBase::ComputeDiv(double & MaxDeltaV, double & normdiv, double & velocityNorm, double & maxdiv,
                const Eigen::VectorXd &u, const Eigen::VectorXd &v, const Eigen::VectorXd &w){

    double div=0,tempdiv=0,tempvelocity=0,uu,vv,ww,md,MaxDeltaV2=0,DeltaV;
    double idx,idy,idz;

    int i,j,k;
    idx=1/dx;
    idy=1/dy;
    idz=1/dz;

//~ Assembling matrix
        #pragma omp parallel for default(shared) private(i,j,k,div,uu,vv,ww,DeltaV) reduction(+:tempdiv,tempvelocity), reduction(max:md,MaxDeltaV2)

        // Traverse the image
        for (i=0;i<pindex.shape()(0);i++){
            for (j=0;j<pindex.shape()(1);j++){
                for (k=0;k<pindex.shape()(2);k++){

                    if (pindex(i,j,k)>-1){   // Node found

                        div=0;
                        uu=0;
                        vv=0;
                        ww=0;
                        DeltaV=0;

                        // In the bulk
                        if(i>0){                        //direction x-
                            if (pindex(i-1,j,k)>-1 && uindex(i,j,k)>-1){

                                div+=-u(uindex(i,j,k))*idx;
                                DeltaV+=-u(uindex(i,j,k));
                                uu+=u(uindex(i,j,k))*0.5;

                            }
                        }
                        if (i<pindex.shape()(0)-1){       //direction x+
                            if (pindex(i+1,j,k)>-1 && uindex(i+1,j,k)>-1){

                                div+=u(uindex(i+1,j,k))*idx;
                                DeltaV+=u(uindex(i+1,j,k));
                                uu+=u(uindex(i+1,j,k))*0.5;

                            }
                        }
                        if(j>0){                        //direction y-
                            if (pindex(i,j-1,k)>-1 && vindex(i,j,k)>-1){

                                div+=-v(vindex(i,j,k))*idy;
                                DeltaV+=-v(vindex(i,j,k));
                                vv+=v(vindex(i,j,k))*0.5;

                            }
                        }
                        if (j<pindex.shape()(1)-1){       //direction y+
                            if (pindex(i,j+1,k)>-1 && vindex(i,j+1,k)>-1){

                                div+=v(vindex(i,j+1,k))*idy;
                                DeltaV+=v(vindex(i,j+1,k));
                                vv+=v(vindex(i,j+1,k))*0.5;

                            }
                        }
                        if(k>0 && nz>0){                        //direction z-
                            if (pindex(i,j,k-1)>-1 && windex(i,j,k)>-1){

                                div+=-w(windex(i,j,k))*idz;
                                DeltaV+=-w(windex(i,j,k));
                                ww+=w(windex(i,j,k))*0.5;

                            }
                        }
                        if (k<pindex.shape()(2)-1 && nz>0){       //direction z+
                            if (pindex(i,j,k+1)>-1 && windex(i,j,k+1)>-1){

                                div+=w(windex(i,j,k+1))*idz;
                                DeltaV+=w(windex(i,j,k+1));
                                ww+=w(windex(i,j,k+1))*0.5;

                            }
                        }

                        // and now the boundary conditions...

                        if (i==0 && uindex(i,j,k)>-1){  //direction x-

                            div+= -u(uindex(i,j,k))*idx;
                            DeltaV+= -u(uindex(i,j,k));
                            uu+=u(uindex(i,j,k))*0.5;

                        }
                        if (i==pindex.shape()(0)-1 && uindex(i+1,j,k)>-1){  //direction x+

                            div+= u(uindex(i+1,j,k))*idx;
                            DeltaV+= u(uindex(i+1,j,k));
                            uu+=u(uindex(i+1,j,k))*0.5;

                        }
                            // Nothing to add for the other directions, as the velocity is zero

                        DeltaV=std::abs(DeltaV);
                        if (DeltaV > MaxDeltaV2){
                            MaxDeltaV2 = DeltaV;
                        }
                        div=std::abs(div);
                        if (div > md){
                            md = div;
                        }
                        tempdiv+=div*div;
                        tempvelocity+=std::sqrt(uu*uu+vv*vv+ww*ww);

                    }//end node found

                }//Inner for loop
            }
        }//outer for loop

    normdiv=std::sqrt(tempdiv);
    velocityNorm=tempvelocity/np;
    maxdiv=md;
    MaxDeltaV=MaxDeltaV2;
}

void NavierStokesBase::CorrectVelocityFieldSIMPLE(
        const Eigen::VectorXd &pc,
        Eigen::VectorXd &u,
        Eigen::VectorXd &v,
        Eigen::VectorXd &w,
        const Eigen::SparseMatrix<double,Eigen::RowMajor> & uA,
        const Eigen::SparseMatrix<double,Eigen::RowMajor> & vA,
        const Eigen::SparseMatrix<double,Eigen::RowMajor> & wA,
        double delta_v_max){

    int i, j, k;
    double delta_v ;
    printf("Correcting Velocity\n");

    #pragma omp parallel for default(shared) private(i,j,k,delta_v)

    // Traverse the image
    for (i=0;i<pindex.shape()(0);i++){
        for (j=0;j<pindex.shape()(1);j++){
            for (k=0;k<pindex.shape()(2);k++){

                if (pindex(i,j,k)>-1){   // Node found

                    // In the bulk
                    if(i>0){                        //direction x-
                        if (pindex(i-1,j,k)>-1 && uindex(i,j,k)>-1){
                            delta_v = -(pc(pindex(i-1,j,k)) - pc(pindex(i,j,k)))/(  uA.coeff(uindex(i,j,k),uindex(i,j,k))  *dx);
                            if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                                delta_v = - delta_v_max;
                            }else if (delta_v > 0 && delta_v > delta_v_max){
                                delta_v = delta_v_max;
                            }
                            u(uindex(i,j,k))+= delta_v ;
                        }

                    }
                    if(j>0){                        //direction y-
                        if (pindex(i,j-1,k)>-1 && vindex(i,j,k)>-1){
                            delta_v = -(pc(pindex(i,j-1,k)) - pc(pindex(i,j,k)))/(  vA.coeff(vindex(i,j,k),vindex(i,j,k))  *dy);
                            if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                                delta_v = - delta_v_max;
                            }else if (delta_v > 0 && delta_v > delta_v_max){
                                delta_v = delta_v_max;
                            }
                            v(vindex(i,j,k))+= delta_v;
                        }
                    }

                    if(k>0){                        //direction z-
                        if (pindex(i,j,k-1)>-1 && windex(i,j,k)>-1){
                            delta_v = -(pc(pindex(i,j,k-1)) - pc(pindex(i,j,k)))/(  wA.coeff(windex(i,j,k),windex(i,j,k)) *dz);
                            if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                                delta_v = - delta_v_max;
                            }else if (delta_v > 0 && delta_v > delta_v_max){
                                delta_v = delta_v_max;
                            }
                            w(windex(i,j,k))+=  delta_v;
                        }
                    }

                    // and now the boundary conditions...

                    if (i==0 && uindex(i,j,k)>-1){
                        delta_v = ( pc(pindex(i,j,k)))/( uA.coeff(uindex(i,j,k),uindex(i,j,k)) *dx);
                        if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                            delta_v = - delta_v_max;
                        }else if (delta_v > 0 && delta_v > delta_v_max){
                            delta_v = delta_v_max;
                        }
                        u(uindex(i,j,k)) += delta_v;

                    }
                    if (i==pindex.shape()(0)-1 && uindex(i+1,j,k)>-1){
                        delta_v = -( pc(pindex(i,j,k)))/( uA.coeff(uindex(i+1,j,k),uindex(i+1,j,k)) *dx);
                        if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                            delta_v = - delta_v_max;
                        }else if (delta_v > 0 && delta_v > delta_v_max){
                            delta_v = delta_v_max;
                        }
                        u(uindex(i+1,j,k)) += delta_v;

                    }
                        // Nothing to add for the other directions, as the velocity components are zero.

                }//end node found

            }//Inner for loop
        }
    }//outer for loop

}

void NavierStokesBase::CorrectVelocityFieldSIMPLEC(
        const Eigen::VectorXd &pc, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &w,
        const Eigen::SparseMatrix<double,Eigen::RowMajor> & uA,
        const Eigen::SparseMatrix<double,Eigen::RowMajor> & vA,
        const Eigen::SparseMatrix<double,Eigen::RowMajor> & wA,
        double delta_v_max){

    int i, j, k;
    double delta_v;


    printf("Correcting Velocity\n");

    #pragma omp parallel for default(shared) private(i,j,k,delta_v)

    // Traverse the image
    for (i=0;i<pindex.shape()(0);i++){
        for (j=0;j<pindex.shape()(1);j++){
            for (k=0;k<pindex.shape()(2);k++){

                if (pindex(i,j,k)>-1){   // Node found

                    // In the bulk
                    if(i>0){                        //direction x-
                        if (pindex(i-1,j,k)>-1 && uindex(i,j,k)>-1){
                            delta_v = -(pc(pindex(i-1,j,k)) - pc(pindex(i,j,k)))/( ( uA.row(uindex(i,j,k)).sum()  )*dx);
                            if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                                delta_v = - delta_v_max;
                            }else if (delta_v > 0 && delta_v > delta_v_max){
                                delta_v = delta_v_max;
                            }
                            u(uindex(i,j,k)) += delta_v;
                        }

                    }
                    if(j>0){                        //direction y-
                        if (pindex(i,j-1,k)>-1 && vindex(i,j,k)>-1){
                            delta_v = -(pc(pindex(i,j-1,k)) - pc(pindex(i,j,k)))/( ( vA.row(vindex(i,j,k)).sum()  )*dy);
                            if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                                delta_v = - delta_v_max;
                            }else if (delta_v > 0 && delta_v > delta_v_max){
                                delta_v = delta_v_max;
                            }
                            v(vindex(i,j,k)) += delta_v;
                        }
                    }

                    if(k>0){                        //direction z-
                        if (pindex(i,j,k-1)>-1 && windex(i,j,k)>-1){
                            delta_v = -(pc(pindex(i,j,k-1)) - pc(pindex(i,j,k)))/( ( wA.row(windex(i,j,k)).sum() )*dz);
                            if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                                delta_v = - delta_v_max;
                            }else if (delta_v > 0 && delta_v > delta_v_max){
                                delta_v = delta_v_max;
                            }
                            w(windex(i,j,k)) += delta_v;
                        }
                    }

                    // and now the boundary conditions...

                    if (i==0 && uindex(i,j,k)>-1){
                        delta_v = ( pc(pindex(i,j,k)))/(( uA.row(uindex(i,j,k)).sum() )*dx);
                        if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                            delta_v = - delta_v_max;
                        }else if (delta_v > 0 && delta_v > delta_v_max){
                            delta_v = delta_v_max;
                        }
                        u(uindex(i,j,k)) += delta_v;

                    }
                    if (i==pindex.shape()(0)-1 && uindex(i+1,j,k)>-1){
                        delta_v = - ( pc(pindex(i,j,k)))/(( uA.row(uindex(i+1,j,k)).sum() )*dx);
                        if (delta_v < 0 && std::abs(delta_v) > delta_v_max){
                            delta_v = - delta_v_max;
                        }else if (delta_v > 0 && delta_v > delta_v_max){
                            delta_v = delta_v_max;
                        }
                        u(uindex(i+1,j,k)) += delta_v;

                    }
                        // Nothing to add for the other directions, as the velocity components are zero.

                }//end node found

            }//Inner for loop
        }
    }//outer for loop

}

void NavierStokesBase::CorrectVelocityFieldSIMPLER(
        const Eigen::VectorXd &p,
        Eigen::VectorXd &uc,
        Eigen::VectorXd &vc,
        Eigen::VectorXd &wc,
        const Eigen::VectorXd &u_hat,
        const Eigen::VectorXd &v_hat,
        const Eigen::VectorXd &w_hat,
        const Eigen::SparseMatrix<double,Eigen::RowMajor> & uA,
        const Eigen::SparseMatrix<double,Eigen::RowMajor> & vA,
        const Eigen::SparseMatrix<double,Eigen::RowMajor> & wA,
        double p_0, double p_1){

        // Note: p_0 and p_1 can be set to 0 when p is a correction. Otherwise they must be set to the boundary condition values.

    int i, j, k;

    printf("Correcting Velocity\n");

    #pragma omp parallel for default(shared) private(i,j,k)

    // Traverse the image
    for (i=0;i<pindex.shape()(0);i++){
        for (j=0;j<pindex.shape()(1);j++){
            for (k=0;k<pindex.shape()(2);k++){

                if (pindex(i,j,k)>-1){   // Node found

                    // In the bulk
                    if(i>0){                        //direction x-
                        if (pindex(i-1,j,k)>-1 && uindex(i,j,k)>-1){
                            //~ printf("%d, %d, %d, %f, %f \n", i,j,k, pc(pindex(i-1,j,k)) - pc(pindex(i,j,k)), uA.row(uindex(i,j,k)));
                            uc(uindex(i,j,k)) = u_hat(uindex(i,j,k)) - (p(pindex(i-1,j,k)) - p(pindex(i,j,k)))/(  uA.coeff(uindex(i,j,k),uindex(i,j,k))  *dx);
                        }

                    }
                    if(j>0){                        //direction y-
                        if (pindex(i,j-1,k)>-1 && vindex(i,j,k)>-1){
                            //~ printf("%d, %d, %d, %f, %f \n", i,j,k, p(pindex(i,j-1,k)) - p(pindex(i,j,k)),vA.row(vindex(i,j,k)));
                            vc(vindex(i,j,k)) = v_hat(vindex(i,j,k)) -(p(pindex(i,j-1,k)) - p(pindex(i,j,k)))/(  vA.coeff(vindex(i,j,k),vindex(i,j,k))  *dy);
                        }
                    }

                    if(k>0){                        //direction z-
                        if (pindex(i,j,k-1)>-1 && windex(i,j,k)>-1){
                            //~ printf("%d, %d, %d, %f, %f \n", i,j,k, p(pindex(i,j,k-1)) - p(pindex(i,j,k)), wA.coeff(windex(i,j,k)));
                            wc(windex(i,j,k)) = w_hat(windex(i,j,k)) - (p(pindex(i,j,k-1)) - p(pindex(i,j,k)))/(  wA.coeff(windex(i,j,k),windex(i,j,k)) *dz);
                        }
                    }

                    // and now the boundary conditions...

                    if (i==0 && uindex(i,j,k)>-1){

                        uc(uindex(i,j,k)) = u_hat(uindex(i,j,k)) - ( p_0 - p(pindex(i,j,k)))/( uA.coeff(uindex(i,j,k),uindex(i,j,k)) *dx);

                    }
                    if (i==pindex.shape()(0)-1 && uindex(i+1,j,k)>-1){

                        uc(uindex(i+1,j,k)) = u_hat(uindex(i+1,j,k)) - (p(pindex(i,j,k))- p_1)/( uA.coeff(uindex(i+1,j,k),uindex(i+1,j,k)) *dx);

                    }
                        // Nothing to add for the other directions, as the velocity components are zero.

                }//end node found

            }//Inner for loop
        }
    }//outer for loop

}

void NavierStokesBase::ComputePressureCorrection(Eigen::VectorXd &pc, Eigen::VectorXd &b){

    printf("Solving for pressure correction: ");
    info_file<<"Solving for pressure correction : ";
    psolver.setTolerance(pcsolvertol);
    //pc=psolver.solve(b);
    pc=psolver.solveWithGuess(b,pc);
    //~ pc*=beta;

    if(psolver.info()==Eigen::Success) {
        printf("%d iterations (status %d)\n", (int)psolver.iterations(), (int)psolver.info());
        info_file<<psolver.iterations()<<" iterations"<<" (status "<< psolver.info()<<")"<<std::endl;
    }else{

        if(psolver.info()==Eigen::NumericalIssue){
              printf("Numerical Issue (iterations: %d). ", (int)psolver.iterations());
             info_file<<"Numerical Issue (iterations:"<<psolver.iterations()<<"). ";
        }else if (psolver.info()==Eigen::NoConvergence){
            printf("No convergence (iterations: %d). ", (int)psolver.iterations());
            info_file<<"No convergence (iterations:"<<psolver.iterations()<<"). ";
        }else if (psolver.info()==Eigen::InvalidInput){
            printf("Invalid input (iterations: %d). ", (int)psolver.iterations());
            info_file<<"Invalid input (iterations:"<<psolver.iterations()<<"). ";
        }
        printf("Residual: %g\n", psolver.error());
        info_file <<"Residual: "<< psolver.error()<<std::endl;
    }

}

void NavierStokesBase::CorrectPressureField(Eigen::VectorXd &pc, Eigen::VectorXd &pvector){

    pcmax=beta*pc.maxCoeff();
    pcmin=beta*pc.minCoeff();
    pcnorm=beta*pc.norm();

    printf("Max and min pressure correction: %g, %g\n", pcmax, pcmin);
    info_file<<"Correction de pression maximale et minimale: "<< pcmax<<", "<< pcmin<<std::endl;
    printf("Norm of the pressure correction: %g\n", pcnorm);
    info_file<<"Norme de la correction de pression : "<< pcnorm <<std::endl;

    pvector+=beta*pc;

}


void NavierStokesBase::ComputePressureField(Eigen::VectorXd &pvector, Eigen::VectorXd &b){

    double tol;
    tol=psolvertol;

    printf("Solving for pressure: ");
    info_file<<"Solving for pressure in : ";
    psolver.setTolerance(tol);
    Eigen::VectorXd ptemp;
    ptemp.resize(np);
    ptemp=psolver.solveWithGuess(b,pvector);
    while (psolver.iterations()==0 && psolver.info()==Eigen::Success && tol>1e-15){
        tol*=0.1;
        psolver.setTolerance(tol);
        ptemp=psolver.solveWithGuess(b,ptemp);
    }

    if (psolver.iterations()>0){
        pvector = ptemp;
    }

    if(psolver.info()==Eigen::Success) {
        printf("%d iterations (status %d)\n", (int)psolver.iterations(), (int)psolver.info());
        info_file<<psolver.iterations()<<" iterations"<<" (status "<< psolver.info()<<")"<<std::endl;
    }else{

        if(psolver.info()==Eigen::NumericalIssue){
              printf("Numerical Issue (iterations: %d). ", (int)psolver.iterations());
             info_file<<"Numerical Issue (iterations:"<<psolver.iterations()<<"). ";
        }else if (psolver.info()==Eigen::NoConvergence){
            printf("No convergence (iterations: %d). ", (int)psolver.iterations());
            info_file<<"No convergence (iterations:"<<psolver.iterations()<<"). ";
        }else if (psolver.info()==Eigen::InvalidInput){
            printf("Invalid input (iterations: %d). ", (int)psolver.iterations());
            info_file<<"Invalid input (iterations:"<<psolver.iterations()<<"). ";
        }
        printf("Residual: %g\n", psolver.error());
        info_file << "Residual: "<< psolver.error()<<std::endl;
    }

}

void NavierStokesBase::AddBorder(blitz::Array<unsigned char,3> &image, int BorderLength){

    blitz::Array<unsigned char,3> temp;

    temp.resize(image.shape()(0)+2*BorderLength,image.shape()(1),image.shape()(2));

    temp = 0;
    int i,j,k;

    #pragma omp parallel for default(shared) private(i,j,k)

    for (i=0;i<image.shape()(0);i++){
            for (j=0;j<image.shape()(1);j++){
                for (k=0;k<image.shape()(2);k++){
                    temp(i+BorderLength,j,k)=image(i,j,k);
                }
            }
        }

    image.resize(temp.shape()(0),temp.shape()(1),temp.shape()(2));
    image = temp;

    printf("Adding a border of %d voxels on each side\n", BorderLength);
}


void NavierStokesBase::Rotation(int direction,blitz::Array<unsigned char,3> &image){


    int i,j,k;

    blitz::Array<unsigned char,3> temp;

    if (direction==2){

        temp.resize(image.shape()(1),image.shape()(0),image.shape()(2));

        #pragma omp parallel for default(shared) private(i,j,k)


        for (i=0;i<image.shape()(0);i++){
            for (j=0;j<image.shape()(1);j++){
                for (k=0;k<image.shape()(2);k++){
                    temp(j,-i+image.shape()(0)-1,k)=image(i,j,k);
                }
            }
        }

        }else if (direction==3){

            temp.resize(image.shape()(2),image.shape()(1),image.shape()(0));

            #pragma omp parallel for default(shared) private(i,j,k)

            for (i=0;i<image.shape()(0);i++){
                for (j=0;j<image.shape()(1);j++){
                    for (k=0;k<image.shape()(2);k++){
                        temp(k,j,i)=image(i,j,k);
                    }
                }
            }

        }

        image.resize(temp.shape());
        image=temp;

    }


void NavierStokesBase::InitIndexArrays(const blitz::Array<unsigned char,3> image){

    int i,j,k;
    bool hasneighbor;

    // Initialization
    np=-1,nu=-1,nv=-1,nw=-1;
    pindex=-1;
    uindex=-1;
    vindex=-1;
    windex=-1;

    // No parallelization here, the indices are computed sequentially...

    for (i=0;i<image.shape()(0);i++){
        for (j=0;j<image.shape()(1);j++){
            for (k=0;k<image.shape()(2);k++){

                if ((int)image(i,j,k)==0){
                    hasneighbor=false;

                    if(j>0 && (int)image(i,j-1,k)==0){
                        nv+=1;
                        vindex(i,j,k)=nv;
                        hasneighbor=true;
                    }
                    if(j<image.shape()(1)-1 && (int)image(i,j+1,k)==0){
                        hasneighbor=true;
                    }
                    if(k>0 && (int)image(i,j,k-1)==0){
                        nw+=1;
                        windex(i,j,k)=nw;
                        hasneighbor=true;
                    }
                    if (nz>0){
                        if(k<image.shape()(2)-1 && (int)image(i,j,k+1)==0){
                            hasneighbor=true;
                        }
                    }

                    if (i==0){
                        nu+=1;
                        uindex(i,j,k)=nu;
                        hasneighbor=true;
                    }
                    if(i>0 && (int)image(i-1,j,k)==0){
                        nu+=1;
                        uindex(i,j,k)=nu;
                        hasneighbor=true;

                    }
                    if (i==image.shape()(0)-1 && (int)image(i,j,k)==0){
                        nu+=1;
                        uindex(i+1,j,k)=nu;
                        hasneighbor=true;
                    }
                    if(i<image.shape()(0)-1 && (int)image(i+1,j,k)==0){
                        hasneighbor=true;

                    }

                    if (hasneighbor){

                        np+=1;
                        pindex(i,j,k)=np;
                        //~ std::cout<<i<<","<<j<<","<<k<<","<<pindex(i,j,k)<<std::endl;
                    }

                }// end of test a(i,j,k)=0
            }
        }
    }

    np+=1;
    nu+=1;
    nv+=1;
    nw+=1;

}

void NavierStokesBase::InitVelocity(Eigen::VectorXd &u,Eigen::VectorXd &v,Eigen::VectorXd &w){

    int i,j,k;

    if (vxfilename != "none"){

        blitz::Array<double,3> ufield;
        readvtkfile_double(vxfilename,ufield);

        printf("Importing velocity field x from %s\n", vxfilename.c_str());

        #pragma omp parallel for default(shared) private(i,j,k)

        for (i=0;i<uindex.shape()(0);i++){
            for (j=0;j<uindex.shape()(1);j++){
                for (k=0;k<uindex.shape()(2);k++){
                    if (uindex(i,j,k)!=-1){
                        //std::cout<<old_pfield(i,j,k)<<std::endl;
                        u(uindex(i,j,k))=ufield(i,j,k);
                    }
                }
            }
        }
    }

    if (vyfilename != "none"){

        blitz::Array<double,3> vfield;
        readvtkfile_double(vyfilename,vfield);

        printf("Importing velocity field y from %s\n", vyfilename.c_str());

        #pragma omp parallel for default(shared) private(i,j,k)

        for (i=0;i<vindex.shape()(0);i++){
            for (j=0;j<vindex.shape()(1);j++){
                for (k=0;k<vindex.shape()(2);k++){
                    if (vindex(i,j,k)!=-1){
                        //std::cout<<old_pfield(i,j,k)<<std::endl;
                        v(vindex(i,j,k))=vfield(i,j,k);
                    }
                }
            }
        }
    }

    if (vzfilename != "none"){

        blitz::Array<double,3> wfield;
        readvtkfile_double(vzfilename,wfield);

        printf("Importing velocity field z from %s\n", vzfilename.c_str());

        #pragma omp parallel for default(shared) private(i,j,k)

        for (i=0;i<windex.shape()(0);i++){
            for (j=0;j<windex.shape()(1);j++){
                for (k=0;k<windex.shape()(2);k++){
                    if (windex(i,j,k)!=-1){
                        //std::cout<<old_pfield(i,j,k)<<std::endl;
                        w(windex(i,j,k))=wfield(i,j,k);
                    }
                }
            }
        }
    }

}


void NavierStokesBase::Initpvector(std::string & pfieldfilename, Eigen::VectorXd &pvector){

    int i,j,k;

    if (pfieldfilename != "none"){

        blitz::Array<double,3> old_pfield;
        readvtkfile_double(pfieldfilename,old_pfield);

        printf("Importing pressure field %s\n", pfieldfilename.c_str());

        #pragma omp parallel for default(shared) private(i,j,k)

        for (i=0;i<pindex.shape()(0);i++){
            for (j=0;j<pindex.shape()(1);j++){
                for (k=0;k<pindex.shape()(2);k++){
                    if (pindex(i,j,k)!=-1){
                        //std::cout<<old_pfield(i,j,k)<<std::endl;
                        pvector(pindex(i,j,k))=old_pfield(i,j,k);
                    }
                }
            }
        }

    }else{

        #pragma omp parallel for default(shared) private(i,j,k)

        for (i=0;i<pindex.shape()(0);i++){
            for (j=0;j<pindex.shape()(1);j++){
                for (k=0;k<pindex.shape()(2);k++){
                    if (pindex(i,j,k)!=-1){
                        pvector(pindex(i,j,k))=(p1-p0)*((double)i+1.0)/((double) pindex.shape()(0) ) + p0;
                    }
                }
            }
        }
    }

    printf("p0, p1 (Pa)= %g, %g\n", p0, p1);
    info_file<<"p0, p1 (Pa)= "<<p0<<","<<p1<<std::endl;
    printf("Pressure gradient (Pa/m)= %g\n", (p0-p1)/(dx*pindex.shape()(0)));
    info_file<<"Pressure gradient (Pa/m)= "<<(p0-p1)/(dx*pindex.shape()(0))<<std::endl;
    info_file<<"inlet/outlet area (m²)= "<<(dy*dz*uindex.shape()(1)*uindex.shape()(2))<<std::endl;

}

