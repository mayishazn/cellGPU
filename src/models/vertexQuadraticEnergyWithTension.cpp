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
    //printf("Forces computed\n");
    //ArrayHandle<double2> h_f(vertexForces,access_location::host, access_mode::read);
    //ArrayHandle<double2> h_fs(vertexForceSets,access_location::host, access_mode::overwrite);
    /*for (int i = 0; i < 3*Nvertices ; ++i)
        {
        //printf("vertex %i force = (%g,%g)\n",i,h_f.data[i].x,h_f.data[i].y);
        printf("vertex %i force set = (%g,%g)\n",i,h_fs.data[i].x,h_fs.data[i].y);
        };
        */
    };

/*
double2 VertexQuadraticEnergyWithTension::computedtheta_givencellID(int cellid, double2 vcurr)
{
    std::vector<double2> vertexPos;
    std::vector<int> cellTopology;
    std::vector<int> cellVertIndices;

    // Grab data
    ArrayHandle<double2> h_v(vertexPositions, access_location::host, access_mode::read);
    for (int i = 0; i < 2 * Ncells; ++i) {
        vertexPos.push_back(h_v.data[i]);
    }

    ArrayHandle<int> h_cvn(cellVertexNum, access_location::host, access_mode::read);
    for (int i = 0; i < Ncells; ++i) {
        cellTopology.push_back(h_cvn.data[i]);
    }

    ArrayHandle<int> h_cv(cellVertices, access_location::host, access_mode::read);
    for (int i = 0; i < Ncells; ++i) {
        for (int j = 0; j < vertexMax; ++j) {
            cellVertIndices.push_back(h_cv.data[i * vertexMax + j]);
        }
    }
        int inumneighbors = cellTopology[cellid];
        double sumX = 0.0, sumY = 0.0;

        // Calculate the centroid of the cell
        for (int j = 0; j < inumneighbors; ++j) {
            sumX += vertexPos[cellVertIndices[cellid * vertexMax + j]].x;
            sumY += vertexPos[cellVertIndices[cellid * vertexMax + j]].y;
        }
        double xbar = sumX / inumneighbors;
        double ybar = sumY / inumneighbors;

        // Calculate normalized second central moments for the region
        std::vector<double> x(inumneighbors), y(inumneighbors);
    	double sumxisquared = 0.0;
	    double sumxiyi = 0.0;
	    double sumyisquared = 0.0;
	    double2 dtheta; 

        for (int j = 0; j < inumneighbors; ++j) {
            x[j] = vertexPos[cellVertIndices[cellid * vertexMax + j]].x - xbar;
            y[j] = -(vertexPos[cellVertIndices[cellid * vertexMax + j]].y - ybar); // Negative for orientation calculation
	        sumxisquared += x[j] * x[j];
            sumyisquared += y[j] * y[j];
            sumxiyi += x[j] * y[j];    
        }
	
	double sqrtterm = sqrt(sumxisquared*sumxisquared + 4*sumxiyi*sumxiyi - 2*sumxisquared*sumyisquared + sumyisquared*sumyisquared);
	double num1x = sumxisquared*vcurr.y - 2*vcurr.x*sumxiyi - vcurr.y*sumyisquared;
	double num2 = sumxisquared - sumyisquared + sqrtterm; 
	double denom = sqrtterm*(sumxisquared*sumxisquared + 4*sumxiyi*sumxiyi + sumyisquared*(sumyisquared - sqrtterm) + sumxisquared*(-2*sumyisquared+sqrtterm));
	dtheta.x = num1x*num2/denom;

	double num1y = 2*vcurr.y*sumxiyi+vcurr.x*(sumxisquared-sumyisquared);
	dtheta.y = num1y*num2/denom;

    return dtheta;
}; 
*/


double2 VertexQuadraticEnergyWithTension::computedtheta_givencellID(int cellid, double2 vcurr)
{
    std::vector<double2> vertexPos;
    std::vector<int> cellTopology;
    std::vector<int> cellVertIndices;

    // Grab data
    ArrayHandle<double2> h_v(vertexPositions, access_location::host, access_mode::read);
    for (int i = 0; i < 2 * Ncells; ++i) {
        vertexPos.push_back(h_v.data[i]);
    }

    ArrayHandle<int> h_cvn(cellVertexNum, access_location::host, access_mode::read);
    for (int i = 0; i < Ncells; ++i) {
        cellTopology.push_back(h_cvn.data[i]);
    }

    ArrayHandle<int> h_cv(cellVertices, access_location::host, access_mode::read);
    for (int i = 0; i < Ncells; ++i) {
        for (int j = 0; j < vertexMax; ++j) {
            cellVertIndices.push_back(h_cv.data[i * vertexMax + j]);
        }
    }
        int inumneighbors = cellTopology[cellid];
        double sumX = 0.0, sumY = 0.0;

        // Calculate the centroid of the cell
        for (int j = 0; j < inumneighbors; ++j) {
            sumX += vertexPos[cellVertIndices[cellid * vertexMax + j]].x;
            sumY += vertexPos[cellVertIndices[cellid * vertexMax + j]].y;
        }
        double xbar = sumX / inumneighbors;
        double ybar = sumY / inumneighbors;
	double xcurr = vcurr.x - xbar; 
//	double ycurr = -vcurr.y + ybar; 05/10/2025
    double ycurr = vcurr.y - ybar; 

        // Calculate normalized second central moments for the region
        std::vector<double> x(inumneighbors), y(inumneighbors);
    	double uxx = 0.0;
	    double uxy= 0.0;
	    double uyy = 0.0;
	    double avgx = 0.0; 
	    double avgy = 0.0; 
	    double2 dtheta; 

        for (int j = 0; j < inumneighbors; ++j) {
            x[j] = vertexPos[cellVertIndices[cellid * vertexMax + j]].x - xbar;
            y[j] = (vertexPos[cellVertIndices[cellid * vertexMax + j]].y - ybar); // 05/10/2025 changed sign for orientation calculation
	        uxx+= x[j] * x[j];
            uyy+= y[j] * y[j];
            uxy += x[j] * y[j];  
	    avgx += x[j]/inumneighbors; 
	    avgy += y[j]/inumneighbors;
        }

	double duxx_dx, duyy_dx, duxy_dx;
    double duxx_dy, duyy_dy, duxy_dy;

    duxx_dx = 2*(xcurr - avgx);
    duyy_dx = 0.0;
    duxy_dx =  ycurr - avgy; // 05/10/2025 changed sign

    duyy_dy = 2*(ycurr - avgy);
    duxx_dy = 0.0;
    duxy_dy = xcurr - avgx;
 
    double num, den, dnum_dx, dden_dx, dnum_dy, dden_dy; 

    // Compute num and den based on the if statement
    if (uyy > uxx) { //this if statement is for numerical stability
    //essentially you do not want the denominator to be 0
        num = uyy - uxx + std::sqrt((uyy - uxx) * (uyy - uxx) + 4 * uxy * uxy);
        den = 2 * uxy;

        // Derivatives of num and den
        dnum_dx = duyy_dx - duxx_dx +
                  ((uyy - uxx) * (duyy_dx - duxx_dx) + 8 * uxy * duxy_dx) /
                  std::sqrt((uyy - uxx) * (uyy - uxx) + 4 * uxy * uxy);
        dden_dx = 2 * duxy_dx;
        dnum_dy = duyy_dy - duxx_dy +
                  ((uyy - uxx) * (duyy_dy - duxx_dy) + 8 * uxy * duxy_dy) /
                  std::sqrt((uyy - uxx) * (uyy - uxx) + 4 * uxy * uxy);
        dden_dy = 2 * duxy_dy;
    } else {
        num = 2 * uxy;
        den = uxx - uyy + std::sqrt((uxx - uyy) * (uxx - uyy) + 4 * uxy * uxy);

        // Derivatives of num and den
        dnum_dx = 2 * duxy_dx;
        dden_dx = duxx_dx - duyy_dx +
                  ((uxx - uyy) * (duxx_dx - duyy_dx) + 8 * uxy * duxy_dx) /
                  std::sqrt((uxx - uyy) * (uxx - uyy) + 4 * uxy * uxy);
        dnum_dy = 2 * duxy_dy;
        dden_dy = duxx_dy - duyy_dy +
                  ((uxx - uyy) * (duxx_dy - duyy_dy) + 8 * uxy * duxy_dy) /
                  std::sqrt((uxx - uyy) * (uxx - uyy) + 4 * uxy * uxy);
    }

    // Compute derivative of theta

    if (den == 0.0) {
        if (num == 0.0) {
            dtheta.x = dden_dx/pow(10,-10); //some big number
            dtheta.y = dden_dy/pow(10,-10); //some big number
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
    return dtheta;
}; 

void VertexQuadraticEnergyWithTension::computeVertexTensionForcesCPU()
{
    //energy function is E =  1/2 ((P-P0)^2/P0)*(1+alpha*A*N sin^2(theta-actinAngle))
    //where P is the perimeter, P0 is the preferred perimeter, A is the area, N is the number of cells, and theta is the orientation of the cell
    //the x component of the force is then
    //-dE/dx = -(P-P0)/P0 * (1+alpha*A*N sin^2(theta-actinAngle)) dP/dx - (P-P0)^2/P0 * (2*alpha*A*N sin(theta-actinAngle)cos(theta-actinAngle))*dtheta/dx
    
    ArrayHandle<int> h_vcn(vertexCellNeighbors,access_location::host,access_mode::read);
    //std::cout << "Size of vertexCellNeighbors: " << vertexCellNeighbors.getNumElements() << std::endl;
    ArrayHandle<double2> h_vc(voroCur,access_location::host,access_mode::read);
    ArrayHandle<double4> h_vln(voroLastNext,access_location::host,access_mode::read);
    ArrayHandle<double2> h_AP(AreaPeri,access_location::host,access_mode::read);
    ArrayHandle<double2> h_APpref(AreaPeriPreferences,access_location::host,access_mode::read);
    ArrayHandle<int> h_ct(cellType,access_location::host,access_mode::read);
    ArrayHandle<int> h_cv(cellVertices,access_location::host, access_mode::read);
    ArrayHandle<int> h_cvn(cellVertexNum,access_location::host,access_mode::read);
    ArrayHandle<double> h_tm(tensionMatrix,access_location::host,access_mode::read);

    ArrayHandle<double2> h_fs(vertexForceSets,access_location::host, access_mode::overwrite);
    //std::cout << "Size of vertexForces: " << vertexForces.getNumElements() << std::endl;
    ArrayHandle<double2> h_f(vertexForces,access_location::host, access_mode::overwrite);
    //std::cout << "Size of theta: " << theta.getNumElements() << std::endl;
    ArrayHandle<double2> h_theta(theta,access_location::host,access_mode::read);


    int nForceSets = Nvertices*3;
    
    double2 vlast,vcur,vnext;
    double alpha = actinStrength; 
    //std::cout << "Number of forces" << Nvertices*3 << std::endl;
    for(int fsidx = 0; fsidx < Nvertices*3; ++fsidx) { //LOOP THROUGH ALL NEIGHBORS OF ALL VERTICIES
        //printf("fsidx = %i\n",fsidx);
            //if (fsidx == 1)
           // {
            //printf("size of h_vcn = %i\n",h_vcn.data.getNumElements());
           // printf("cellIdx1 = %i\n",h_vcn.data[fsidx]);
           // };
        int cellIdx1 = h_vcn.data[fsidx];
        double Adiff = (KA/2)*(h_AP.data[cellIdx1].x - h_APpref.data[cellIdx1].x);
        //if (fsidx == 1)
          //  {
           // printf("Adiff = %f\n",Adiff);
            //};
        vcur = h_vc.data[fsidx];
        vlast.x = h_vln.data[fsidx].x;  vlast.y = h_vln.data[fsidx].y;
        vnext.x = h_vln.data[fsidx].z;  vnext.y = h_vln.data[fsidx].w;
        //if (fsidx == 1)
         //   {
          //  printf("vcur = (%f,%f), vlast = (%f,%f), vnext = (%f,%f)\n",vcur.x,vcur.y,vlast.x,vlast.y,vnext.x,vnext.y);
           // };
            
       //first lets get dP/dx and dP/dy for each vertex
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
       // if (fsidx == 0)
        //    {
         //   printf("dPdv = (%f,%f)\n",dPdv.x,dPdv.y);
          //  printf("dAdv = (%f,%f)\n",dAdv.x,dAdv.y);
           // printf("dlast = (%f,%f), dnext = (%f,%f)\n",dlast.x,dlast.y,dnext.x,dnext.y);
            //printf("dlnorm = %f, dnnorm = %f\n",dlnorm,dnnorm);
           // printf("Adiff = %f\n",Adiff);
           // };
        h_fs.data[fsidx].x = 2.0*(Adiff*dAdv.x);
        h_fs.data[fsidx].y = 2.0*(Adiff*dAdv.y);
   
        double2 dtheta = computedtheta_givencellID(cellIdx1, vcur);
        //if (fsidx == 1)
         //   {
          //  printf("dtheta = (%f,%f)\n",dtheta.x,dtheta.y);
           // printf("|dtheta.x| > 0: (%d)\n",abs(dtheta.x) > 0);
           // printf("|dtheta.y| > 0: (%d)\n",abs(dtheta.y) > 0);
           // };
       //energy function is E =  1/2 ((P-P0)^2/P0)*(1+alpha*A*N sin^2(theta-actinAngle))
       //Now compute first term --> note 
       //-(P-P0)/P0 * (1+alpha*A*N sin^2(theta-actinAngle)) dP/dx == termA * dP/dx
       double termA = KP*(h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y)/h_APpref.data[cellIdx1].y * (1+alpha*sin(h_theta.data[cellIdx1].x - h_theta.data[cellIdx1].y)*sin(h_theta.data[cellIdx1].x - h_theta.data[cellIdx1].y));


       //Now compute second term
       //-(P-P0)^2/P0 * (2*alpha*A*N sin(theta-actinAngle)cos(theta-actinAngle))*dtheta/dx == termB * dtheta/dx
       double termB = -KP*(h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y)*(h_AP.data[cellIdx1].y - h_APpref.data[cellIdx1].y)/h_APpref.data[cellIdx1].y * (2*alpha*sin(h_theta.data[cellIdx1].x - h_theta.data[cellIdx1].y)*cos(h_theta.data[cellIdx1].x - h_theta.data[cellIdx1].y));
        double termA_x = termA * dPdv.x;
        double termA_y = termA * dPdv.y;
        double termB_x = termB * dtheta.x;
        double termB_y = termB * dtheta.y;

        h_fs.data[fsidx].x += termA_x + termB_x; // 05/10/2025 changed sign
        h_fs.data[fsidx].y += termA_y + termB_y; // 05/10/2025 changed sign
       //Now add the adhesion term
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
            if (!std::isfinite(h_fs.data[fsidx].x)||!std::isfinite(h_fs.data[fsidx].y))//||h_fs.data[fsidx].x>100||h_fs.data[fsidx].y>100)
            {
                printf("fsidx = %i\n",fsidx); //print all the things here
                printf("cellIdx1 = %i\n",h_vcn.data[fsidx]);
                printf("Adiff = %f\n",Adiff);
                printf("termA = %f, termB = %f\n",termA,termB);
                printf("dPdv = (%f,%f), dtheta = (%f,%f)\n",dPdv.x,dPdv.y,dtheta.x,dtheta.y);
                printf("vcur = (%f,%f), vlast = (%f,%f), vnext = (%f,%f)\n",vcur.x,vcur.y,vlast.x,vlast.y,vnext.x,vnext.y);
                printf("dAdv = (%f,%f)\n",dAdv.x,dAdv.y);
                printf("dlast = (%f,%f), dnext = (%f,%f)\n",dlast.x,dlast.y,dnext.x,dnext.y);
                printf("dlnorm = %f, dnnorm = %f\n",dlnorm,dnnorm); 
                printf("dtheta = (%f,%f)\n",dtheta.x,dtheta.y);
                printf("|dtheta.x| > 0: (%d)\n",abs(dtheta.x) > 0);
                printf("|dtheta.y| > 0: (%d)\n",abs(dtheta.y) > 0);
                printf("Perimeter pref: %f",h_APpref.data[fsidx].y); 
                printf("finite perimeter preference? %d",std::isfinite(h_APpref.data[fsidx].y));
                printf("sin cos term: %f", sin(h_theta.data[fsidx].x - h_theta.data[fsidx].y)*cos(h_theta.data[fsidx].x - h_theta.data[fsidx].y));
                printf("sin term: %f", sin(h_theta.data[fsidx].x - h_theta.data[fsidx].y));
                printf("cos term: %f", cos(h_theta.data[fsidx].x - h_theta.data[fsidx].y));
                
    for (int i = 0; i < Ncells; ++i) {
        printf("Cell %i: Area = %f, Perimeter = %f\n", i, h_AP.data[i].x, h_AP.data[i].y);
    }
        for (int i = 0; i < Ncells; ++i) {
        printf("Cell %i: Area Preference = %f, Perimeter Preference = %f\n", i, h_APpref.data[i].x, h_APpref.data[i].y);
    }
        for (int i = 0; i < Ncells; ++i) {
        printf("Cell %i: theta = %f, actinangle = %f\n", i, h_theta.data[i].x, h_theta.data[i].y);
    }
            };
            if(std::isnan(h_fs.data[fsidx].x)||std::isnan(h_fs.data[fsidx].y))
            {
                printf("fsidx = %i\n",fsidx); //print all the things here
                printf("cellIdx1 = %i\n",h_vcn.data[fsidx]);
                printf("Adiff = %f\n",Adiff);
                printf("termA = %f, termB = %f\n",termA,termB);
                printf("dPdv = (%f,%f), dtheta = (%f,%f)\n",dPdv.x,dPdv.y,dtheta.x,dtheta.y);
                printf("vcur = (%f,%f), vlast = (%f,%f), vnext = (%f,%f)\n",vcur.x,vcur.y,vlast.x,vlast.y,vnext.x,vnext.y);
                printf("dAdv = (%f,%f)\n",dAdv.x,dAdv.y);
                printf("dlast = (%f,%f), dnext = (%f,%f)\n",dlast.x,dlast.y,dnext.x,dnext.y);
                printf("dlnorm = %f, dnnorm = %f\n",dlnorm,dnnorm); 
                printf("dtheta = (%f,%f)\n",dtheta.x,dtheta.y);
                printf("|dtheta.x| > 0: (%d)\n",abs(dtheta.x) > 0);
                printf("|dtheta.y| > 0: (%d)\n",abs(dtheta.y) > 0);
                printf("Perimeter pref: %f",h_APpref.data[fsidx].y); 
                printf("finite perimeter preference? %d",std::isfinite(h_APpref.data[fsidx].y));
                printf("sin term: %f", sin(h_theta.data[fsidx].x - h_theta.data[fsidx].y));
                printf("cos term: %f", cos(h_theta.data[fsidx].x - h_theta.data[fsidx].y));
                
    for (int i = 0; i < Ncells; ++i) {
        printf("Cell %i: Area = %f, Perimeter = %f\n", i, h_AP.data[i].x, h_AP.data[i].y);
    }
        for (int i = 0; i < Ncells; ++i) {
        printf("Cell %i: Area Preference = %f, Perimeter Preference = %f\n", i, h_APpref.data[i].x, h_APpref.data[i].y);
    }
        for (int i = 0; i < Ncells; ++i) {
        printf("Cell %i: theta = %f, actinangle = %f\n", i, h_theta.data[i].x, h_theta.data[i].y);
    }
            };
     };//end of loop through each vertex

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
    ArrayHandle<double2> h_theta(theta,access_location::host,access_mode::read);
    double alpha = actinStrength; 

    // Initialize the sum
    double sum = 0.0;
    int count = 0; // To count the number of valid terms

    // Loop over all cells
    for (int i = 0; i < Ncells; ++i) {
        double P_i = h_AP.data[i].y; // Perimeter of cell i
        double P_i0 = h_APpref.data[i].y; // Perimeter preference of cell i
        double theta_i = h_theta.data[i].x; // Orientation of cell i
        double theta_i0 = h_theta.data[i].y; // Actin angle of cell i
        //perimeters[i] = 2 * std::sqrt(M_PI * areas[i])
        // Loop over neighbors of cell i
        for (int j : cellNeighbors[i]) {
            double P_j = h_AP.data[j].y; // Perimeter of cell j
            double P_j0 = h_APpref.data[j].y; // Perimeter preference of cell j
            double theta_j = h_theta.data[j].x; // Orientation of cell j
            double theta_j0 = h_theta.data[j].y; // Actin angle of cell j
            // Compute the terms
            
            double termA = KP*(P_i - P_i0)/P_i0 * (1+alpha*sin(theta_i - theta_i0)*sin(theta_i - theta_i0));
            double termB = -KP*(P_i - P_i0)*(P_i - P_i0)/P_i0 * (2*alpha*sin(theta_i - theta_i0)*cos(theta_i - theta_i0));
    
            double term_i = termA+termB;

            
            termA = KP*(P_j - P_j0)/P_j0 * (1+alpha*sin(theta_j - theta_j0)*sin(theta_j - theta_j0));
            termB = -KP*(P_j - P_j0)*(P_j - P_j0)/P_j0 * (2*alpha*sin(theta_j - theta_j0)*cos(theta_j - theta_j0));
            double term_j = termA+termB;

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
    //std::cout << "count = " << count << std::endl;
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
    ArrayHandle<double2> d_v(vertexPositions,access_location::device,access_mode::read);
    ArrayHandle<double2> d_AP(AreaPeri,access_location::device,access_mode::read);
    ArrayHandle<double2> d_APpref(AreaPeriPreferences,access_location::device,access_mode::read);
    ArrayHandle<int> d_ct(cellType,access_location::device,access_mode::read);
    ArrayHandle<int> d_cv(cellVertices,access_location::device, access_mode::read);
    ArrayHandle<int> d_cvn(cellVertexNum,access_location::device,access_mode::read);
    ArrayHandle<double> d_tm(tensionMatrix,access_location::device,access_mode::read);
    ArrayHandle<double2> d_theta(theta,access_location::device,access_mode::read);
    ArrayHandle<double2> d_fs(vertexForceSets,access_location::device, access_mode::overwrite);
    ArrayHandle<double2> d_f(vertexForces,access_location::device, access_mode::overwrite);

    int nForceSets = Nvertices*3;
    gpu_vertexModel_tension_force_sets(
            d_vcn.data,
            d_vc.data,
            d_vln.data,
            d_v.data,
            d_AP.data,
            d_APpref.data,
            d_theta.data, 
            d_ct.data,
            d_cv.data,
            d_cvn.data,
            d_tm.data,
            d_fs.data,
            cellTypeIndexer,
            n_idx,
            simpleTension,
            gamma,
            actinStrength, 
            nForceSets,
            vertexMax,
            Ncells,
            KA,KP
            );

            /**/

    gpu_avm_sum_force_sets(d_fs.data, d_f.data,Nvertices);
    };

