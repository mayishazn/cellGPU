#ifndef VertexQuadraticEnergyWithTension_H
#define VertexQuadraticEnergyWithTension_H

#include "vertexQuadraticEnergy.h"

/*! \file vertexQuadraticEnergyWithTension.h */
//!Add line tension terms between different "types" of cells in the 2D Vertex model
/*!
This child class of VertexQuadraticEnergy is completely analogous to the voversion on the Voronoi side.
It implements different tension terms between different types of cells, and different routines are
called depending on whether multiple different cell-cell surface tension values are needed.
 */
class VertexQuadraticEnergyWithTension : public VertexQuadraticEnergy
    {
    public:
        //! initialize with random positions and set all cells to have uniform target A_0 and P_0 parameters
        VertexQuadraticEnergyWithTension(int n, double A0, double P0,bool reprod = false, bool runSPVToInitialize=false,bool usegpu = true) : VertexQuadraticEnergy(n,A0,P0,reprod,runSPVToInitialize,usegpu)
                {
                gamma = 0;Tension = false;simpleTension = true;GPUcompute = usegpu; forceExponent=1.0;
                if(!GPUcompute)
                    tensionMatrix.neverGPU =true;
                };

        //!compute the geometry and get the forces
        virtual void computeForces();
        //!compute the quadratic energy functional
        virtual double computeEnergy();

        //!Compute the net force on particle i on the CPU with multiple tension values
        virtual void computeVertexTensionForcesCPU();
        //!call gpu_force_sets kernel caller
        virtual void computeVertexTensionForceGPU();

        //!Use surface tension
        void setUseSurfaceTension(bool use_tension){Tension = use_tension;};
        //!Set surface tension, with only a single value of surface tension
        void setSurfaceTension(double g){gamma = g; simpleTension = true;};
        //!Set a general flattened 2d matrix describing surface tensions between many cell types
        void setSurfaceTension(vector<double> gammas);
        //!Get surface tension
        double getSurfaceTension(){return gamma;};
        void setForceExponent(double x){forceExponent = x;};

        double reportMeanEdgeTension();
    protected:
        //!The value of surface tension between two cells of different type
        double gamma;
        //exponent of force
        double forceExponent;
        //!A flag specifying whether the force calculation contains any surface tensions to compute
        bool Tension;
        //!A flag switching between "simple" tensions (only a single value of gamma for every unlike interaction) or not
        bool simpleTension;
        //!A flattened 2d matrix describing the surface tension, \gamma_{i,j} for types i and j
        GPUArray<double> tensionMatrix;

    //be friends with the associated Database class so it can access data to store or read
    friend class AVMDatabaseNetCDF;
    };

#endif
