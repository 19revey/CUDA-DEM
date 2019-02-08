/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL, float smallParticleRadius, float largeParticleRadius, float gap);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
			ACCELERATION,
			MASS,
			VELOCITYM,
			ANGULARVEL,
			ANGULARVELM,
			ANGULARACL,
			COLOR,
			CONTACTS
        };

        void update(float deltaTime);
        void reset(ParticleConfig config);

        float *getArray(ParticleArray array);
        void   setArray(ParticleArray array, const float *data, int start, int count);

        int    getNumParticles() const
        {
            return m_numParticles;
        }
		bool getStatus()
		{
			return m_bReady;
		}
		void switchStatus()
		{
			m_bReady = !m_bReady;
		}
		unsigned int getStepCount()
		{
			return stepCount;
		}
        unsigned int getCurrentReadBuffer() const
        {
            return m_posVbo;
        }
        unsigned int getColorBuffer()       const
        {
            return m_colorVBO;
        }
		unsigned int getVelocityBuffer()	const
		{
			return m_velVbo;
		}
        void *getCudaPosVBO()              const
        {
            return (void *)m_cudaPosVBO;
        }
        void *getCudaColorVBO()            const
        {
            return (void *)m_cudaColorVBO;
        }

        void dumpGrid();
		void dumpNeighborLists();
        void dumpParticles(uint start, uint count);

        void setIterations(int i)
        {
            m_solverIterations = i;
        }
        void setDamping(float x)
        {
            m_params.globalDamping = x;
        }
        void setGravity(float x)
        {
            m_params.gravity = make_float3(0.0f, x, 0.0f);
        }
        void setCollideNSpring(float x)
        {
            m_params.nspring = x;
        }
		void setCollideTSpring(float x)
		{
			m_params.tspring = x;
		}
        void setCollideNDamping(float x)
        {
            m_params.ndamping = x;
        }
        void setCollideTDamping(float x)
        {
            m_params.tdamping = x;
        }
		void setCollideShear(float x)
		{
			m_params.cshear = x;
		}
		void setColoumbStaticFric(float x)
		{
			m_params.coloumb = x;
		}
        void setCollideAttraction(float x)
        {
            m_params.attraction = x;
        }
        void setCollideRollingFriction(float x)
        {
        	m_params.rollfc = x;
        }
        void setBoundaryRollingFriction(float x)
        {
        	m_params.rollfb = x;
        }
		void setReadPosition(int t)
		{
			m_readPos = t;
		}
		void setBoundaryShear(float x)
		{
			m_params.bshear = x;
		}
		void setBoundaryColoumb(float x)
		{
			m_params.bcoloumb = x;
		}
		void setBoundarySpring(float x)
		{
			m_params.bspring = x;
		}
		void setBoundaryDamping(float x)
		{
			m_params.bdamping = x;
		}
		void setFeedOn(bool flag)
		{
			if (flag == 0) {
				m_bFeedOn = 1;
			}
			else {
				m_bFeedOn = 0;
			}
		}
		void setRecordStep(uint x)
		{
			m_recordStep = x;
		}
		bool getFeedOn()
		{
			return m_bFeedOn;
		}
        float getParticleRadius()
        {
            return m_params.particleRadius;
        }
        uint3 getGridSize()
        {
            return m_params.gridSize;
        }
		int getCurrentTotal()
		{
			return m_ntotal;
		}
		void setCurrentTotal(int newTotal)
		{
			m_ntotal = newTotal;
		}
        float3 getWorldOrigin()
        {
            return m_params.worldOrigin;
        }
        float3 getCellSize()
        {
            return m_params.cellSize;
        }
		void changeWriteState()
		{
			m_bOutput = !m_bOutput;
		}
		void setFeedSpacing(uint3 spacing)
		{
			m_feedSpacing = spacing;
		}
		void setInitialVelocity(float3 v)
		{
			initVelocity = v;
		}
		void setMaxDropHeight(float h)
		{
			maxDropHeight = h;
		}

        void initContactList();
		void initBCList();
		void initializeDistribution(float number);
		void addParticles();
		void printContacts();
		void openRestartFile(int readPos);
		void write2Bin();

    protected: // methods
        ParticleSystem() {}
        uint createVBO(uint size);

        void _initialize(int numParticles);
        void _finalize();



    protected: // data
        bool m_bInitialized, m_bUseOpenGL, m_bFeedOn, m_bOutput, m_bRestart, m_bReady;
		std::string outputFileName;
        uint m_numParticles;
		int m_ntotal;
		uint stepCount;
		uint backupStepCount;
		float maxDropHeight;
		int fallSteptime;
		float m_gapThickness;
		float3 initVelocity;
		float maxFallTime;
		float smallParticleRadius;
		float largeParticleRadius;
		uint m_recordStep;
		int m_readPos;
		FILE* m_restartFile;
		float m_pvolume;
		uint3 m_feedSpacing;

        // CPU data
		float *m_hRadiiTemp;		// Temp array holding variable number of particle radii to fill desired total volume
		float *m_hRadiiNormalSelection;	// Temp array holding the normally distributed particle radii
		float *m_hRadiiLogSelection;	// Temp array holding the log-normally distributed particle radii

        float *m_hPos;              // particle positions
        float *m_hVel;              // particle velocities
		float *m_hAcl;				// particle accelerations
		float *m_hMass;				// particle masses
		float *m_hCol;				// particle assigned colors
		float *m_hVelm;				// particle intermediate velocities
		float *m_hOmegVel;			// particle angular velocities
		float *m_hOmegVelm;			// particle intermediate angular velocities
		float *m_hOmegAcl;			// particle angular accelerations

		float *m_hCurrentNeighborList;		// particle neighbor list (used for initialization)
		float *m_hNewNeighborList;
		float *m_htangwallfront;
		float *m_htangwallback;
		float *m_htangwallbottom;
		float *m_htangwalltop;
		float *m_htangwallleft;
		float *m_htangwallright;
		float *m_hTotalNormalForce;

		uint *m_hPurgatory; // List of particle numbers that have crossed the right boundary

        uint  *m_hParticleHash;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;
		uint  *m_hNumNeighbors;

		uint  *m_hListHeads;
		uint  *m_hNextPointers;

        // GPU data
        float *m_dPos;
        float *m_dVel;
		float *m_dMass;
		float *m_dVelm;
		float *m_dAcl;
		float *m_dOmegAcl;
		float *m_dOmegVel;
		float *m_dOmegVelm;

		float *m_dCurrentNeighborList;      // 2d linear pitched array containing particle tangdisp data
		float *m_dNewNeighborList;

		float *m_dtangwallfront;
		float *m_dtangwallback;
		float *m_dtangwallleft;
		float *m_dtangwallright;
		float *m_dtangwallbottom;
		float *m_dtangwalltop;
		float *m_dTotalNormalForce;

		uint *m_dPurgatory; // Device-side list of particle numbers that have crossed the right boundary

        float *m_dSortedPos;
        float *m_dSortedVel;
		float *m_dSortedMass;
		float *m_dSortedVelm;
		float *m_dSortedOmegVel;
		float *m_dSortedOmegVelm;
		uint  *m_dSortedNumNeighbors;

        // grid data for sorting method
        uint  *m_dGridParticleHash;		// grid hash value for each particle
        uint  *m_dGridParticleIndex;	// particle index for each particle
		uint  *m_dNumNeighbors;			// array which stores the number of neighbors in each particle's neighbor list
        uint  *m_dCellStart;			// index of start of each cell in sorted list
        uint  *m_dCellEnd;				// index of end of cell

		uint  *m_dListHeads;			// array of first elements in each cell
		uint  *m_dNextPointers;			// array of pointers to the next element in the linked list

        uint   m_gridSortBits;

        uint   m_posVbo;				// vertex buffer object for particle positions
        uint   m_colorVBO;				// vertex buffer object for colors
		uint   m_velVbo;				// vertex buffer object for velocities (used for visualization)

        float *m_cudaPosVBO;			// these are the CUDA deviceMem Pos
        float *m_cudaColorVBO;			// these are the CUDA deviceMem Color
		float *m_cudaVelVBO;			// these are the CUDA deviceMem Velocities

        struct cudaGraphicsResource *m_cuda_posvbo_resource;	// handles OpenGL-CUDA exchange
		struct cudaGraphicsResource *m_cuda_velvbo_resource;	// handles OpenGL-CUDA exchange
        struct cudaGraphicsResource *m_cuda_colorvbo_resource;	// handles OpenGL-CUDA exchange

        // params
        SimParams m_params;
        uint3 m_gridSize;

        uint m_numGridCells;

        StopWatchInterface *m_timer;

        uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
