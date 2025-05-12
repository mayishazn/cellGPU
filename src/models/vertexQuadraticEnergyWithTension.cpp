#include "vertexQuadraticEnergyWithTension.h"
#include "vertexQuadraticEnergyWithTension.cuh"
#include "vertexQuadraticEnergy.cuh"
/*! \file vertexQuadraticEnergyWithTension.cpp */

/*!
This function defines a matrix, \gamma_{i,j}, describing the imposed tension between cell types i and
j. This function both sets that matrix and sets the flag telling the computeForces function to call
the more general tension force computations.
\pre the vector has n^2 elements, where n is the number of types, and the values of type in the system
must be of the form 0, 1, 2, ...n. The input vector should be laid out as:
gammas[0] = g_{0,0}  (an irrelevant value that is never called)
gammas[1] = g_{0,1}
gammas[n] = g_{0,n}
gammas[n+1] = g_{1,0} (physically, this better be the same as g_{0,1})
gammas[n+2] = g_{1,1} (again, never used)
...
gammas[n^2-1] = g_{n,n}
*/
void VertexQuadraticEnergyWithTension::setSurfaceTension(vector<double> gammas)
    {
    simpleTension = false;
    //set the tension matrix to the right size, and the indexer
    tensionMatrix.resize(gammas.size());
    int n = sqrt(gammas.size());
    cellTypeIndexer = Index2D(n);

    ArrayHandle<double> tensions(tensionMatrix,access_location::host,access_mode::overwrite);
    for (int ii = 0; ii < gammas.size(); ++ii)
        {
        int typeI = ii/n;
        int typeJ = ii - typeI*n;
        tensions.data[cellTypeIndexer(typeJ,typeI)] = gammas[ii];
        };
    };

/*!
goes through the process of computing the forces on either the CPU or GPU, either with or without
exclusions, as determined by the flags. Assumes the geometry has NOT yet been computed.
\post the geometry is computed, and force per cell is computed.
*/
void VertexQuadraticEnergyWithTension::computeForces()
    {
    if(forcesUpToDate)
       return; 
    forcesUpToDate = true;
    computeGeometry();
    if (GPUcompute)
        {
        if (Tension)
            computeVertexTensionForceGPU();
        else
            computeForcesGPU();
        }
    else
        {
        if(Tension)
                computeVertexTensionForcesCPU();
        else
            computeForcesCPU();
        };
    };

/*!
Use the data pre-computed in the geometry routine to rapidly compute the net force on each vertex...for the cpu part combine the simple and complex tension routines
*/
void VertexQuadraticEnergyWithTension::computeVertexTensionForcesCPU()
    {
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    ArrayHandle<double2> h_vc(voroCur,access_location::host,access_mode::read);
    ArrayHandle<double4> h_vln(voroLastNext,access_location::host,access_mode::read);
    ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<double2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<double> h_tm(tensionMatrix,access_location::host,access_mode::read);

    ArrayHandle<double2> h_fs(vertexForceSets,access_location::host, access_mode::overwrite);
    ArrayHandle<double2> h_f(vertexForces,access_location::host, access_mode::overwrite);

    //first, compute the contribution to the force on each vertex from each of its three cells
    double2 vlast,vcur,vnext;
    double2 dEdv;
    double Adiff, Pdiff;
    for(int fsidx = 0; fsidx < Nvertices*3; ++fsidx)
        {
        //for the change in the energy of the cell, just repeat the vertexQuadraticEnergy part
        int cellIdx1 = h_vcn.data[fsidx];
        double Adiff = KA/2*h_APpref.data[cellIdx1].x*pow((h_AP.data[cellIdx1].x - h_APpref.data[cellIdx1].x)/h_APpref.data[cellIdx1].x,1);
        //double Adiff = 0; //it seems I have to pass in Adiff to computeForceSetVertexModel 
        double Pdiff = KP/2*(forceExponent+1)*pow(abs(h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y)/h_APpref.data[cellIdx1].y,forceExponent)*(h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y)/abs((h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y));
        vcur = h_vc.data[fsidx];
        vlast.x = h_vln.data[fsidx].x;  vlast.y = h_vln.data[fsidx].y;
        vnext.x = h_vln.data[fsidx].z;  vnext.y = h_vln.data[fsidx].w;

        //computeForceSetVertexModel is defined in inc/utility/functions.h
        //computeForceSetVertexModel(vcur,vlast,vnext,Adiff,Pdiff,dEdv);
        //(const double2 &vcur, const double2 &vlast, const double2 &vnext,
        //const double &Adiff, const double &Pdiff,
        //double2 &dEdv)
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
        //end replacement snippet
        h_fs.data[fsidx].x = dEdv.x;
        h_fs.data[fsidx].y = dEdv.y;

        //first, determine the index of the cell other than cellIdx1 that contains both vcur and vnext
        int cellNeighs = h_cvn.data[cellIdx1];
        //find the index of vcur and vnext
        int vCurIdx = fsidx/3;
        int vNextInt = 0;
        if (h_cv.data[n_idx(cellNeighs-1,cellIdx1)] != vCurIdx)
            {
            for (int nn = 0; nn < cellNeighs-1; ++nn)
                {
                int idx = h_cv.data[n_idx(nn,cellIdx1)];
                if (idx == vCurIdx)
                    vNextInt = nn +1;
                };
            };
        int vNextIdx = h_cv.data[n_idx(vNextInt,cellIdx1)];

        //vcur belongs to three cells... which one isn't cellIdx1 and has both vcur and vnext?
        int cellIdx2 = 0;
        int cellOfSet = fsidx-3*vCurIdx;
        for (int cc = 0; cc < 3; ++cc)
            {
            if (cellOfSet == cc) continue;
            int cell2 = h_vcn.data[3*vCurIdx+cc];
            int cNeighs = h_cvn.data[cell2];
            for (int nn = 0; nn < cNeighs; ++nn)
                if (h_cv.data[n_idx(nn,cell2)] == vNextIdx)
                    cellIdx2 = cell2;
            }
        //now, determine the types of the two relevant cells, and add an extra force if needed
        int cellType1 = h_ct.data[cellIdx1];
        int cellType2 = h_ct.data[cellIdx2];
        if(cellType1 != cellType2)
            {
            double gammaEdge;
            if (simpleTension)
                gammaEdge = gamma;
            else
                gammaEdge = h_tm.data[cellTypeIndexer(cellType1,cellType2)];
            double2 dnext = vcur-vnext;
            double dnnorm = sqrt(dnext.x*dnext.x+dnext.y*dnext.y);
            h_fs.data[fsidx].x -= gammaEdge*dnext.x/dnnorm;
            h_fs.data[fsidx].y -= gammaEdge*dnext.y/dnnorm;
            };
        };

    //now sum these up to get the force on each vertex
    for (int v = 0; v < Nvertices; ++v)
        {
        double2 ftemp = make_double2(0.0,0.0);
        for (int ff = 0; ff < 3; ++ff)
            {
            ftemp.x += h_fs.data[3*v+ff].x;
            ftemp.y += h_fs.data[3*v+ff].y;
            };
        h_f.data[v] = ftemp;
        };
    };

double VertexQuadraticEnergyWithTension::reportMeanEdgeTension()
    {
    if(!forcesUpToDate)
        computeForces();

    std::vector<std::vector<int>> cellNeighbors = reportCellNeighbors();
    ArrayHandle<double2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
    // Initialize the sum
    double sum = 0.0;
    int count = 0; // To count the number of valid terms

    // Loop over all cells
    for (int i = 0; i < Ncells; ++i) {
        double P_i = h_AP.data[i].y; // Perimeter of cell i
        double P_i0 = h_APpref.data[i].y; // Perimeter preference of cell i
        //perimeters[i] = 2 * std::sqrt(M_PI * areas[i])
        // Loop over neighbors of cell i
        for (int j : cellNeighbors[i]) {
            double P_j = h_AP.data[j].y; // Perimeter of cell j
            double P_j0 = h_APpref.data[j].y; // Perimeter preference of cell j
            // Compute the terms
            double term_i = (forceExponent+1)*KP*pow(abs(P_i - P_i0)/P_i0,forceExponent)*((P_i - P_i0)/abs((P_i - P_i0)));
            double term_j = (forceExponent+1)*KP*pow(abs(P_j - P_j0)/P_j0,forceExponent)*((P_j - P_j0)/abs((P_j - P_j0)));

            // Compute the expression
            double expression = term_i + term_j - gamma;

            // Ensure the expression is positive before taking the logarithm
            if (expression > 0) {
                if (expression<1e-9) 
                    {expression = 1e-9;} //testing if setting a small positive number threshold helps
                sum += log10(expression);
                count++; 
            }
            else { //print out min and max neg numbers maybe???
                sum += log10(1e-9);
                count++; 
            }
            
        }
    }
    std::cout << "count = " << count << std::endl;
    // Compute the mean of the logarithm
    double meanLogEdgeTension = (count > 0) ? (sum / count) : 0.0;
    //double meanLogEdgeTension = sum / (3*Ncells);
    return meanLogEdgeTension;
    };


double VertexQuadraticEnergyWithTension::computeEnergy()
    {
    if(!forcesUpToDate)
        computeForces();


    //return totalEnergy;
//};
    printf("computeEnergy function for VertexQuadraticEnergyWithTension not written. Very sorry\n");
    throw std::exception();
    return 0;
    };

void VertexQuadraticEnergyWithTension::computeVertexTensionForceGPU()
    {
    ArrayHandle<int> d_vcn(vertexCellNeighbors,access_location::device,access_mode::read);
    ArrayHandle<double2> d_vc(voroCur,access_location::device,access_mode::read);
    ArrayHandle<double4> d_vln(voroLastNext,access_location::device,access_mode::read);
    ArrayHandle<double2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<double2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<int> d_ct(cellType,access_location::device,access_mode::read);
    ArrayHandle<int> d_cv(cellVertices,access_location::device, access_mode::read);
    ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::read);
    ArrayHandle<double> d_tm(tensionMatrix,access_location::device,access_mode::read);

    ArrayHandle<double2> d_fs(vertexForceSets,access_location::device, access_mode::overwrite);
    ArrayHandle<double2> d_f(vertexForces,access_location::device, access_mode::overwrite);

    int nForceSets = Nvertices*3;
    gpu_vertexModel_tension_force_sets(
            d_vcn.data,
            d_vc.data,
            d_vln.data,
            d_AP.data,
            d_APpref.data,
            d_ct.data,
            d_cv.data,
            d_cvn.data,
            d_tm.data,
            d_fs.data,
            cellTypeIndexer,
            n_idx,
            simpleTension,
            forceExponent,
            gamma,
            nForceSets,
            KA,KP
            );

    gpu_avm_sum_force_sets(d_fs.data, d_f.data,Nvertices);
    };

