#include "std_include.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"


#include "vertexQuadraticEnergyWithTension.h"
#include "EnergyMinimizerFIRE2D.h"
#include "DatabaseNetCDFAVM.h"
#include "logEquilibrationStateWriter.h"
#include "analysisPackage.h"
#include <iostream>
#include <vector>
#include <memory>
#include <ctime>
#include <random>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <numeric>
#include <cmath>

//! A function of convenience for setting FIRE parameters
void setFIREParameters(shared_ptr<EnergyMinimizerFIRE> emin, double deltaT, double alphaStart,
        double deltaTMax, double deltaTInc, double deltaTDec, double alphaDec, int nMin,
        double forceCutoff)
    {
    emin->setDeltaT(deltaT);
    emin->setAlphaStart(alphaStart);
    emin->setDeltaTMax(deltaTMax);
    emin->setDeltaTInc(deltaTInc);
    emin->setDeltaTDec(deltaTDec);
    emin->setAlphaDec(alphaDec);
    emin->setNMin(nMin);
    emin->setForceCutoff(forceCutoff);
    };

// Function to print the contents of a std::vector<std::vector<int>>
void printVectorOfVectors(const std::vector<std::vector<int>>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "Cell " << i << " neighbors: ";
        for (size_t j = 0; j < vec[i].size(); ++j) {
            std::cout << vec[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


double calculate_mean(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0; // Handle empty data case
    }

    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double calculate_std_dev(const std::vector<double>& data) {
    if (data.size() < 2) {
        return 0.0; // Handle cases with insufficient data
    }

    double mean = calculate_mean(data);
    double sum_sq_diff = 0.0;

    for (const double& value : data) {
        sum_sq_diff += pow(value - mean, 2);
    }

    return sqrt(sum_sq_diff / (data.size())); //  standard deviation
}
/*!
This file compiles to produce an executable that can be used to reproduce the timing information
for the 2D AVM model found in the "cellGPU" paper, using the following parameters:
i = 1000
t = 4000
e = 0.01
dr = 1.0,
along with a range of v0 and p0. This program also demonstrates the use of brownian dynamics
applied to the vertices themselves.
NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will 
move the positions. If you want the forces and the positions to be sync'ed, you should call the
vertex model's computeForces() funciton right before saving a state.
*/
int main(int argc, char*argv[])
{
    double alpha=2.0;
    // Sweep through adhesion parameters from 0 to 0.3 with an increment of 0.01
    double adhesion = 0.0;
    clock_t t1, t2; // clocks for timing informatio
    int numpts =50; //number of cells
    int USE_GPU =0; //0 or greater uses a gpu, any negative number runs on the cpu
    int tSteps = 100; //number of time steps to run after initialization
    int initSteps = 1000; //number of initialization steps
    double p0 = 4.0;  //the preferred perimeter -> not used
    double NT = 0.4; 
    int Nc = round(NT*numpts);
    double KA = 10.0;
    double KP = 1.0;
    //double alpha = 1.0; //actin alpha not fire alpha
    double a0 = 1.0;  // the preferred area -> not used
    //FIRE Parameters
    double dt = 0.01/KA; //the (initial) time step size
    double alphaStart = 0.99; //firealpha also max alpha
    double deltaTMax = 0.05;  // maximum time step size
    double deltaTInc= 1.01; // time step size increase factor
    double deltaTDec = 0.99; // time step size decrease factor
    double alphaDec = 0.9; // fire alpha decrease factor
    double nMin = 4; // minimum number of iterations before increasing time step size
    double forceCutoff = 1e-13; 
    //End FIRE PARAMETERS
    printf("numpts = %d\n, tSteps = %d\n, initSteps = %d\n", numpts, tSteps, initSteps);
    printf("dt = %f\n, alphaStart = %f\n, deltaTMax = %f\n, deltaTInc = %f\n, deltaTDec = %f\n, alphaDec = %f\n, nMin = %f\n, forceCutoff = %e\n", dt, alphaStart, deltaTMax, deltaTInc, deltaTDec, alphaDec, nMin, forceCutoff);
    double v0 = 0.00;  // the self-propulsion
    double Dr = 0.0;  //the rotational diffusion constant of the cell directors not used
    int program_switch = 0; //various settings control output
    int imgstep = 1; //how often to save images

    int c;
    //generate areas and perimeters for the cells
    std::random_device rd;
    std::mt19937 gen(rd());
    double Ap, bs, k, theta, sum; //parameters for the cell preferences
    double areas[numpts];
    double perimeters[numpts];
    Ap = 0.4; 
    bs = std::sqrt(numpts); 
    k = std::pow(1/Ap,2); 
    theta = std::pow(Ap/bs,2);
    std::gamma_distribution<> d(k,theta);
    // <random> has a gamma distribution
    sum = 0; 
    for (int i=0; i<numpts; ++i)
        {
        areas[i] = d(gen);
        sum += areas[i];
        }
    double newsum = 0; 
    for (int i=0; i<numpts; ++i)
        {
        areas[i] /= (sum/numpts);
        perimeters[i] = 2 * std::sqrt(M_PI * areas[i]);
        newsum += areas[i];
        }
        char buffer[100];
        std::sprintf(buffer, "areas%dp%d.txt", static_cast<int>(std::floor(Ap)), static_cast<int>(std::fmod(std::floor(Ap * 1000), 1000)));
        std::ofstream areaFile(buffer);
        for (int i = 0; i < numpts; ++i) {
            areaFile << areas[i] << "\n";
        }
        areaFile.close();

    while((c=getopt(argc,argv,"n:g:m:s:r:a:i:v:b:x:y:z:p:t:e:d:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'z': program_switch = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case 'd': Dr = atof(optarg); break;
            case '?':
                    if(optopt=='c')
                        std::cerr<<"Option -" << optopt << "requires an argument.\n";
                    else if(isprint(optopt))
                        std::cerr<<"Unknown option '-" << optopt << "'.\n";
                    else
                        std::cerr << "Unknown option character.\n";
                    return 1;
            default:
                       abort();
        };
    bool reproducible = false; // if you want random numbers with a more random seed each run, set this to false
    //check to see if we should run on a GPU
    bool initializeGPU = true;
    bool gpu = chooseGPU(USE_GPU);
    if (!gpu) 
        initializeGPU = false;

    //possibly save output in netCDF format
    //char dataname[256];
        int Nvert = 2*numpts;
    

    bool runSPV = false;//setting this to true will relax the random cell positions to something more uniform before running vertex model dynamics
    ofstream myfile; 

    // Open a file stream to append results
    std::ofstream resultsFile("adhesion_vs_meanEdgeTension.txt", std::ios::app);
    if (!resultsFile.is_open()) {
        std::cerr << "Unable to open file for writing" << std::endl;
        return 1;
    }

    // Open a file stream to append results
    std::ofstream cnFile("adhesion_vs_cn.txt", std::ios::app);
    if (!cnFile.is_open()) {
        std::cerr << "Unable to open file for writing" << std::endl;
        return 1;
    }

    // Open a file stream to append results
    std::ofstream eccentricityFile("adhesion_vs_e.txt", std::ios::app);
    if (!cnFile.is_open()) {
        std::cerr << "Unable to open file for writing" << std::endl;
        return 1;
    }

    //for (double alpha=1.0; alpha >= 0; alpha -= 0.1) {
    // Sweep through adhesion parameters from 0 to 0.3 with an increment of 0.01
    //for (double adhesion = 0.0; adhesion <= 1.00; adhesion += 0.01) {
        // Initialize the vertex model
    
    // Initialize the system with vertex positions
    shared_ptr<VertexQuadraticEnergyWithTension> vertexModel = make_shared<VertexQuadraticEnergyWithTension>(numpts, 1.0/numpts, 2.3094/numpts, reproducible, initializeGPU);
    vector<int> types(numpts,0);
    vector<double> theta0(numpts,1.57079632679);
        for (int ii = 0; ii < numpts; ++ii)
        {
        types[ii]=ii; //each cell type need be different for the tension to be applied
        }
    vertexModel->setCellType(types);
    vertexModel->setSurfaceTension(adhesion);
    vertexModel->setActinStrength(alpha);
    vertexModel->setUseSurfaceTension(true); //set the flag to use the surface tension
    vertexModel->setModuliUniform(KA, KP); //first number is KA and second is KP
    vertexModel->setCellTheta(theta0); 
    std::vector<int> cellneighbors = vertexModel->reportCellNeighborCounts();

    /*
    //set a certain percentage of cells to have areas that go as their topologies
    std::vector<int> topind;
    for(int i=0; i<Nc; i++) topind.push_back(1);
    for(int i=Nc; i<numpts; i++) topind.push_back(0);
    std::random_shuffle(topind.begin(),topind.end());

    std::vector<int> inei;
    std::vector<int> inds;
    for (int i=0;i<cellneighbors.size();i++) inds.push_back(i);
    //This is a bad sorting algorithm, do not judge me
    while(inds.size()){
    int lowest=0;
    for (int i=0;i<inds.size();i++) if(cellneighbors[inds[lowest]]>cellneighbors[inds[i]]) lowest=i;
    inei.push_back(lowest);
    inds.erase(inds.begin()+lowest);
    }
    
    std::vector<float> rarea, sarea;
    for(int i=0; i<numpts; i++) sarea.push_back(areas[i]);

    std::sort(sarea.begin(),sarea.end());
    for(int i=0; i<numpts; i++){
    if(topind[i]) areas[inei[i]]=sarea[i];
    else rarea.push_back(sarea[i]);
    }
    std::random_shuffle(rarea.begin(),rarea.end());
    for (int i=0; i<numpts; i++) if(!topind[i]) areas[inei[i]]=rarea[i];
    */
    //set cell areas and perimeters to array defined via gamma distribution
    std::vector<double2> AreaPeriPreferences(numpts);
    for (int i = 0; i < numpts; ++i) {
        AreaPeriPreferences[i] = {areas[i], perimeters[i]};
    }

        // Save initial cell neighbors

        std::sprintf(buffer, "initialneighbors_g%dp%d.txt", static_cast<int>(std::floor(adhesion)), static_cast<int>(std::fmod(std::floor(adhesion * 1000), 1000)));
        std::ofstream initialNeighborsFile(buffer);
        for (int i = 0; i < numpts; ++i) {
            initialNeighborsFile << cellneighbors[i] << "\n" << std::endl;
        }
        initialNeighborsFile.close();

        // Save area preferences
        std::sprintf(buffer, "areapreference_g%dp%d.txt", static_cast<int>(std::floor(adhesion)), static_cast<int>(std::fmod(std::floor(adhesion * 1000), 1000)));
        std::ofstream areaPreferenceFile(buffer);
        for (int i = 0; i < numpts; ++i) {
            areaPreferenceFile << areas[i] << "\n";
        }
        areaPreferenceFile.close();

        // Save perimeter preferences
        std::sprintf(buffer, "perimeterpreference_g%dp%d.txt", static_cast<int>(std::floor(adhesion)), static_cast<int>(std::fmod(std::floor(adhesion * 1000), 1000)));
        std::ofstream perimeterPreferenceFile(buffer);
        for (int i = 0; i < numpts; ++i) {
            perimeterPreferenceFile << perimeters[i] << "\n";
        }
        perimeterPreferenceFile.close();

    vertexModel->setCellPreferences(AreaPeriPreferences);
    //when an edge gets less than this long, perform a simple T1 transition
    vertexModel->setT1Threshold(0.01);

    // Set the initial positions to be within [0, 1] for both x and y directions
    vertexModel->setRectangularUnitCell(sqrt(newsum),sqrt(newsum));

    
    // Use the FIRE algorithm to find a minimal energy state
    shared_ptr<EnergyMinimizerFIRE> fireMinimizer = make_shared<EnergyMinimizerFIRE>(vertexModel);

    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(vertexModel);
    sim->addUpdater(fireMinimizer,vertexModel);
    sim->setIntegrationTimestep(dt);
//        if(initSteps > 0)
//           sim->setSortPeriod(initSteps/10);
        //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);

    vertexModel->computeGeometry();
    printf("minimized value of q = %f\n",vertexModel->reportq());
    double meanQ = vertexModel->reportq();
    double varQ = vertexModel->reportVarq();
    double2 variances = vertexModel->reportVarAP();
    printf("Cell <q> = %f\t Var(p) = %g\n",meanQ,variances.y);

        // Update the ncdat file name to reflect the adhesion parameter
        std::sprintf(buffer, "tissuesim_g%dp%d_Ap%dp%d_NT%dp%d_alpha%dp%d.nc", 
                     static_cast<int>(std::floor(adhesion)), 
                     static_cast<int>(std::fmod(std::floor(adhesion * 1000), 1000)), 
                     static_cast<int>(std::floor(Ap)), 
                     static_cast<int>(std::fmod(std::floor(Ap * 1000), 1000)), 
                     static_cast<int>(std::floor(NT)), 
                     static_cast<int>(std::fmod(std::floor(NT * 1000), 1000)),
                     static_cast<int>(std::floor(alpha)),
                     static_cast<int>(std::fmod(std::floor(alpha * 1000), 1000)));

        std::string dataname(buffer);

        AVMDatabaseNetCDF ncdat(Nvert,dataname,NcFile::Replace);
        //save the initial state
        ncdat.writeState(vertexModel);
        // Run the simulation
        for (int i = 0; i < initSteps; ++i) {
            //void setFIREParameters(shared_ptr<EnergyMinimizerFIRE> emin, double deltaT, double alphaStart,
       // double deltaTMax, double deltaTInc, double deltaTDec, double alphaDec, int nMin,
       // double forceCutoff)

            // Set the FIRE parameters
            setFIREParameters(fireMinimizer, dt, alphaStart, deltaTMax, deltaTInc, deltaTDec, alphaDec, nMin, forceCutoff);
            //setFIREParameters(fireMinimizer, dt, 0.99, 0.05, 1.1, 0.95, 0.9, 4, 1e-13);
            fireMinimizer->setMaximumIterations(tSteps * (i + 1));
            sim->performTimestep();
            if (i % imgstep == 0) {
                ncdat.writeState(vertexModel);
            }
            double mf = fireMinimizer->getMaxForce();
            double tens = vertexModel->reportMeanEdgeTension();
            if (mf < 1e-12) {
                break;
                ncdat.writeState(vertexModel);
            }
            if (tens < -5) {
                break;
                ncdat.writeState(vertexModel);
            }
        }

        vertexModel->computeGeometry();
        double meanEdgeTension = vertexModel->reportMeanEdgeTension();
        printf("Adhesion: %f, alpha %f,  Mean log edge tension: %f\n", adhesion, alpha, meanEdgeTension);
        // Append the adhesion and resulting meanEdgeTension to the text file
        resultsFile << adhesion << "\t" << alpha << "\t" << meanEdgeTension << "\n";

        // compute width of distribution of number of neighbors
        std::vector<int> numneighs = vertexModel->reportCellNeighborCounts();
        std::vector<double> doubleVec;
        doubleVec.reserve(numneighs.size()); // Reserve space to avoid multiple allocations
        for (int val : numneighs) {
        doubleVec.push_back(static_cast<double>(val));
        }
        double cn = calculate_std_dev(doubleVec);
        cn = cn*cn;
        cnFile << adhesion << "\t" << cn << "\n";

        std::vector<vector<double>> stats = vertexModel->calculateregionprops();
        double sumEccentricity = 0.0;
        int numCells = stats.size();

        for (const auto& cell : stats) {
         sumEccentricity += cell[4]; // Assuming eccentricity is at index 4
        }

        double meane = numCells > 0 ? sumEccentricity / numCells : 0.0;
        eccentricityFile << adhesion << "\t" << meane << "\n";

    

    resultsFile.close();
    if(initializeGPU)
        cudaDeviceReset();

    return 0;  
}; 
