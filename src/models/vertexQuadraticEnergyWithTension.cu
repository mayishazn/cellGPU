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
            double2 *areaPeri,
            double2 *APPref,
            int *cellType,
            int *cellVertices,
            int *cellVertexNum,
            double *tensionMatrix,
            double2 *forceSets,
            Index2D cellTypeIndexer,
            Index2D n_idx,
            bool simpleTension,
            double forceExponent,
            double gamma,
            int nForceSets,
            double KA, double KP)
    
    {
    unsigned int fsidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (fsidx >= nForceSets)
        return;
    
    //
    //first, compute the geometrical part of the force set using pre-computed data
    //
    double2 vlast,vcur,vnext,dEdv;

    int cellIdx1 = vertexCellNeighbors[fsidx];
    double Adiff = KA/2*(areaPeri[cellIdx1].x - APPref[cellIdx1].x);
    double Pdiff = KP/2*(forceExponent+1)*pow(abs(areaPeri[cellIdx1].y/APPref[cellIdx1].y-1),forceExponent)*(areaPeri[cellIdx1].y - APPref[cellIdx1].y)/abs((areaPeri[cellIdx1].y - APPref[cellIdx1].y));
    //printf("cellIdx1 %d Adiff %f Pdiff %f \n",cellIdx1,Adiff,Pdiff);
    //printf("cellIdx1 %d, P-P0=%f, Pdiff=%f, forceExponent=%f \n",cellIdx1,areaPeri[cellIdx1].y - APPref[cellIdx1].y,Pdiff,forceExponent);
    //if (areaPeri[cellIdx1].y == areaPeri[cellIdx1].y)
    //{printf("cellIdx1 %d, P=%f, P0=%f, forceExponent=%f \n",cellIdx1,areaPeri[cellIdx1].y, APPref[cellIdx1].y,forceExponent);
    //}
    //double Adiff = KA*(areaPeri[cellIdx1].x - APPref[cellIdx1].x);
    //double Pdiff = KP*(areaPeri[cellIdx1].y - APPref[cellIdx1].y);
    vcur = voroCur[fsidx];
    vlast.x = voroLastNext[fsidx].x;  vlast.y = voroLastNext[fsidx].y;
    vnext.x = voroLastNext[fsidx].z;  vnext.y = voroLastNext[fsidx].w;

    //computeForceSetVertexModel(vcur,vlast,vnext,Adiff,Pdiff,dEdv);
    //replacement for computeForceSetVertexModel
    double2 dlast,dnext,dAdv,dPdv;

    //note that my conventions for dAdv and dPdv take care of the minus sign, so
    //that dEdv below is reall -dEdv, so it's the force
    dAdv.x = 0.5*(vlast.y-vnext.y); //half distance between neighboring points
    dAdv.y = -0.5*(vlast.x-vnext.x);
    dlast.x = vlast.x-vcur.x;
    dlast.y = vlast.y-vcur.y;
    double dlnorm = sqrt(dlast.x*dlast.x+dlast.y*dlast.y);
    dnext.x = vcur.x-vnext.x;
    dnext.y = vcur.y-vnext.y;
    double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
    dPdv.x = dlast.x/dlnorm - dnext.x/dnnorm;
    dPdv.y = dlast.y/dlnorm - dnext.y/dnnorm;
        
    //compute the area of the triangle to know if it is positive (convex cell) or not
    //    double TriAreaTimes2 = -vnext.x*vlast.y+vcur.y*(vnext.x-vlast.x)+vcur.x*(vlast.y-vnext.x)+vlast.x+vnext.y;
    //    double TriAreaTimes2 = dlast.x*dnext.y - dlast.y*dnext.x;
    dEdv.x = (Adiff*dAdv.x + Pdiff*dPdv.x);
    dEdv.y = (Adiff*dAdv.y + Pdiff*dPdv.y);
    //dEdv.x = (2)*(Adiff*dAdv.x + Pdiff*dPdv.x);
    //dEdv.y = (2)*(Adiff*dAdv.y + Pdiff*dPdv.y);
    //end replacement snippet
    forceSets[fsidx].x = dEdv.x;
    forceSets[fsidx].y = dEdv.y;

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
    };

bool gpu_vertexModel_tension_force_sets(
        int *vertexCellNeighbors,
        double2 *voroCur,
        double4 *voroLastNext,
        double2 *areaPeri,
        double2 *APPref,
        int *cellType,
        int *cellVertices,
        int *cellVertexNum,
        double *tensionMatrix,
        double2 *forceSets,
        Index2D &cellTypeIndexer,
        Index2D &n_idx,
        bool simpleTension,
        double forceExponent,
        double gamma,
        int nForceSets,
        double KA, double KP)
{
    unsigned int block_size = 128;
    if (nForceSets < 128) block_size = 32;
    unsigned int nblocks  = nForceSets/block_size + 1;

    vm_tensionForceSets_kernel<<<nblocks,block_size>>>(
            vertexCellNeighbors,voroCur,
            voroLastNext,areaPeri,APPref,
            cellType,cellVertices,cellVertexNum,
            tensionMatrix,forceSets,cellTypeIndexer,
            n_idx,simpleTension, forceExponent, gamma,
            nForceSets,KA,KP
            );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
};
/** @} */ //end of group declaration
