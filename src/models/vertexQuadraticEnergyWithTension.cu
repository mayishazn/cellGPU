#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "vertexQuadraticEnergyWithTension.cuh"

/** \file vertexQuadraticEnergyWithTension.cu
    * Defines kernel callers and kernels for GPU calculations of vertex model parts
*/

/*!
    \addtogroup vmKernels
    @{
*/

__global__ void vm_tensionForceSets_kernel(
            int *vertexCellNeighbors,
            double2 *voroCur,
            double4 *voroLastNext,
            double2 *vertexPositions,
            double2 *areaPeri,
            double2 *APPref,
            double2 *theta, 
            int *cellType,
            int *cellVertices,
            int *cellVertexNum,
            double *tensionMatrix,
            double2 *forceSets,
            Index2D cellTypeIndexer,
            Index2D n_idx,
            bool simpleTension,
            double gamma,
            double actinStrength, 
            int nForceSets,
            int vertexMax,
            int Ncells,
            double KA, double KP)
    
    {
    unsigned int fsidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (fsidx >= nForceSets)
        return;
    
    //
    //first, compute the geometrical part of the force set using pre-computed data
    //
    double2 vlast,vcur,vnext;

    int cellIdx1 = vertexCellNeighbors[fsidx];
    double Adiff = KA/2*(areaPeri[cellIdx1].x - APPref[cellIdx1].x);
    double alpha = actinStrength; 
    //double Pdiff = KP*(areaPeri[cellIdx1].y - APPref[cellIdx1].y);
    vcur = voroCur[fsidx];
    vlast.x = voroLastNext[fsidx].x;  vlast.y = voroLastNext[fsidx].y;
    vnext.x = voroLastNext[fsidx].z;  vnext.y = voroLastNext[fsidx].w;
    double2 dlast,dnext,dAdv,dPdv;
    dAdv.x = 0.5*(vlast.y-vnext.y);
    dAdv.y = -0.5*(vlast.x-vnext.x);
    dlast.x = vlast.x-vcur.x;
    dlast.y = vlast.y-vcur.y;
    double dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
    dnext.x = vcur.x-vnext.x;
    dnext.y = vcur.y-vnext.y;
    double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
    dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
    dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;
        
    //computeForceSetVertexModel(vcur,vlast,vnext,Adiff,Pdiff,dEdv);
    forceSets[fsidx].x = 2.0*Adiff*dAdv.x;
    forceSets[fsidx].y = 2.0*Adiff*dAdv.y;

    //Now, to the potential for tension terms...
    //first, determine the index of the cell other than cellIdx1 that contains both vcur and vnext
    int cellNeighs = cellVertexNum[cellIdx1];
    //find the index of vcur and vnext
    int vCurIdx = fsidx/3;
    int vNextInt = 0;
    if (cellVertices[n_idx(cellNeighs-1,cellIdx1)] != vCurIdx)
        {
        for (int nn = 0; nn < cellNeighs-1; ++nn)
            {
            int idx = cellVertices[n_idx(nn,cellIdx1)];
            if (idx == vCurIdx)
                vNextInt = nn +1;
            };
        };
    int vNextIdx = cellVertices[n_idx(vNextInt,cellIdx1)];

    //vcur belongs to three cells... which one isn't cellIdx1 and has both vcur and vnext?
    int cellIdx2 = 0;
    int cellOfSet = fsidx-3*vCurIdx;
    for (int cc = 0; cc < 3; ++cc)
        {
        if (cellOfSet == cc) continue;
        int cell2 = vertexCellNeighbors[3*vCurIdx+cc];
        int cNeighs = vertexCellNeighbors[cell2];
        for (int nn = 0; nn < cNeighs; ++nn)
            if (cellVertices[n_idx(nn,cell2)] == vNextIdx)
                cellIdx2 = cell2;
        }
    //now, determine the types of the two relevant cells, and add an extra force if needed
    int cellType1 = cellType[cellIdx1];
    int cellType2 = cellType[cellIdx2];
    if(cellType1 != cellType2)
        {
        double gammaEdge;
        if (simpleTension)
            gammaEdge = gamma;
        else
            gammaEdge = tensionMatrix[cellTypeIndexer(cellType1,cellType2)];
        double2 dnext = vcur-vnext;
        double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
        forceSets[fsidx].x -= gammaEdge*dnext.x/dnnorm;
        forceSets[fsidx].y -= gammaEdge*dnext.y/dnnorm;
        };
    
    //double2 dtheta = computedtheta_givencellID(cellIdx1, vcur);
    int inumneighbors = cellVertexNum[cellIdx1];
    double sumX = 0.0;
    double sumY = 0.0;
    // Calculate the centroid of the cell
        for (int j = 0; j < inumneighbors; ++j) {
            sumX += vertexPositions[cellVertices[cellIdx1 * vertexMax + j]].x;
            sumY += vertexPositions[cellVertices[cellIdx1 * vertexMax + j]].y;
        }
        double xbar = sumX / inumneighbors;
        double ybar = sumY / inumneighbors;
        double xcurr = vcur.x - xbar;
        double ycurr = vcur.y - ybar; // 05/10/2025
        // Calculate normalized second central moments for the region

        double uxx = 0.0;
        double uxy= 0.0;
        double uyy = 0.0;
        double avgx = 0.0; 
        double avgy = 0.0; 
        double2 dtheta; 
    
            for (int j = 0; j < inumneighbors; ++j) {
                double x = vertexPositions[cellVertices[cellIdx1 * vertexMax + j]].x - xbar;
                double y = (vertexPositions[cellVertices[cellIdx1 * vertexMax + j]].y - ybar); // // 05/10/2025
                uxx+= x * x;
                uyy+= y * y;
                uxy += x * y;  
                avgx += x/inumneighbors; 
                avgy += y/inumneighbors;
            }
    
	
            double duxx_dx, duyy_dx, duxy_dx;
            double duxx_dy, duyy_dy, duxy_dy;
        
            duxx_dx = 2*(xcurr - avgx);
            duyy_dx = 0.0;
            duxy_dx = ycurr - avgy; // 05/10/2025
        
            duyy_dy = 2*(ycurr - avgy);
            duxx_dy = 0.0;
            duxy_dy = xcurr - avgx;
         
            double num, den, dnum_dx, dden_dx, dnum_dy, dden_dy; 
        
            // Compute num and den based on the if statement
            if (uyy > uxx) { //this if statement is for numerical stability
            //essentially you do not want the denominator to be 0
                num = uyy - uxx + sqrt((uyy - uxx) * (uyy - uxx) + 4 * uxy * uxy);
                den = 2 * uxy;
        
                // Derivatives of num and den
                dnum_dx = duyy_dx - duxx_dx +
                          ((uyy - uxx) * (duyy_dx - duxx_dx) + 8 * uxy * duxy_dx) /
                          sqrt((uyy - uxx) * (uyy - uxx) + 4 * uxy * uxy);
                dden_dx = 2 * duxy_dx;
                dnum_dy = duyy_dy - duxx_dy +
                          ((uyy - uxx) * (duyy_dy - duxx_dy) + 8 * uxy * duxy_dy) /
                          sqrt((uyy - uxx) * (uyy - uxx) + 4 * uxy * uxy);
                dden_dy = 2 * duxy_dy;
            } else {
                num = 2 * uxy;
                den = uxx - uyy + sqrt((uxx - uyy) * (uxx - uyy) + 4 * uxy * uxy);
        
                // Derivatives of num and den
                dnum_dx = 2 * duxy_dx;
                dden_dx = duxx_dx - duyy_dx +
                          ((uxx - uyy) * (duxx_dx - duyy_dx) + 8 * uxy * duxy_dx) /
                          sqrt((uxx - uyy) * (uxx - uyy) + 4 * uxy * uxy);
                dnum_dy = 2 * duxy_dy;
                dden_dy = duxx_dy - duyy_dy +
                          ((uxx - uyy) * (duxx_dy - duyy_dy) + 8 * uxy * duxy_dy) /
                          sqrt((uxx - uyy) * (uxx - uyy) + 4 * uxy * uxy);
            }
        
            // Compute derivative of theta
        
            if (den == 0.0) {
                if (num == 0.0) {
                    dtheta.x = -dden_dx/pow(10,-10); //some big number
                    dtheta.y = -dden_dy/pow(10,-10); //some big number
                } else {
                    dtheta.x = -dden_dx/num;  
                    dtheta.y = -dden_dy/num; 
                }
            }
            else {
                double quotient_x = (dnum_dx * den - num * dden_dx) / (den * den);
                double quotient_y = (dnum_dy * den - num * dden_dy) / (den * den);
                dtheta.x =  quotient_x / (1 + (num / den) * (num / den));
                dtheta.y =  quotient_y / (1 + (num / den) * (num / den));
            }



    //energy function is E =  1/2 ((P-P0)^2/P0)*(1+alpha*A*N sin^2(theta-actinAngle))
    //Now compute first term --> note 
    //-(P-P0)/P0 * (1+alpha*A*N sin^2(theta-actinAngle)) dP/dx == termA * dP/dx
    //trying without the AN piece for now. N likely blew this thing up. 
    double termA = KP*(areaPeri[cellIdx1].y - APPref[cellIdx1].y)/APPref[cellIdx1].y * (1+alpha*sin(theta[cellIdx1].x - theta[fsidx].y)*sin(theta[cellIdx1].x - theta[cellIdx1].y));


    //Now compute second term
    //-(P-P0)^2/P0 * (2*alpha*A*N sin(theta-actinAngle)cos(theta-actinAngle))*dtheta/dx == termB * dtheta/dx
    double termB = -KP*(areaPeri[cellIdx1].y - APPref[cellIdx1].y)*(areaPeri[cellIdx1].y - APPref[cellIdx1].y)/areaPeri[cellIdx1].y * (2*alpha*sin(theta[cellIdx1].x - theta[cellIdx1].y)*cos(theta[cellIdx1].x - theta[cellIdx1].y));
    
    double termA_x = termA * dPdv.x;
    double termA_y = termA * dPdv.y;
    double termB_x = termB * dtheta.x;
    double termB_y = termB * dtheta.y;
    forceSets[fsidx].x += termA_x + termB_x;
    forceSets[fsidx].y += termA_y + termB_y;
 
    };

bool gpu_vertexModel_tension_force_sets(
        int *vertexCellNeighbors,
        double2 *voroCur,
        double4 *voroLastNext,
        double2 *vertexPositions,
        double2 *areaPeri,
        double2 *APPref,
        double2 *theta, 
        int *cellType,
        int *cellVertices,
        int *cellVertexNum,
        double *tensionMatrix,
        double2 *forceSets,
        Index2D &cellTypeIndexer,
        Index2D &n_idx,
        bool simpleTension,
        double gamma,
        double actinStrength,
        int nForceSets,
        int vertexMax,
        int Ncells,
        double KA, double KP)
{
    unsigned int block_size = 128;
    if (nForceSets < 128) block_size = 32;
    unsigned int nblocks  = nForceSets/block_size + 1;

    vm_tensionForceSets_kernel<<<nblocks,block_size>>>(
            vertexCellNeighbors,voroCur,
            voroLastNext, vertexPositions, areaPeri,APPref, theta,
            cellType,cellVertices,cellVertexNum,
            tensionMatrix,forceSets,cellTypeIndexer,
            n_idx,simpleTension,gamma,actinStrength,
            nForceSets, vertexMax, Ncells, KA,KP
            );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
};
/** @} */ //end of group declaration
