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

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/iterator/permutation_iterator.h"
#include "thrust/scatter.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/copy.h"
#include "thrust/device_new.h"
#include <time.h>


#define RHO_L 3000.0f
#define RHO_H 12000.0f
#define ASD 40000.0f

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#define NUM_CONTACTS		32
float   topWallPos= 0.8f; //was 0.045f //initial height of top wall, in meters
float   topWallVel = 0.0f;
float   topWallMass = (RHO_L+RHO_H)/2*0.6*0.01*0.2816*0.0352;//0.1f*0.34315f;// 4.0f* (2.0f+1.0f)/2.0f * 0.34315; //2.0f * 5.0f / 2.0f * (0.49078f/2.0f*64.0f/48.0f); //kg; equal to some value times the total weight of the particle bed; divided by two b/c I halved bed height; multiplied by 5/2 to make up for heavy particles with 4x density;
float   topWallAcl = 0.0f;
float shearRate=25.0f;
#endif



ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL, float smallRadius, float largeRadius, float gap) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
	m_bFeedOn(1),
	m_bOutput(1),
	m_bReady(true),
	m_bRestart(false),
	m_restartFile(NULL),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
	m_hAcl(0),
	m_hMass(0),
	m_hCol(0),
	m_hOmegVel(0),
	m_hOmegVelm(0),
	m_hVelm(0),
	m_hOmegAcl(0),
	m_hCurrentNeighborList(0),
	m_hNewNeighborList(0),
	m_hNumNeighbors(0),
    m_dPos(0),
    m_dVel(0),
	m_dMass(0),
	m_dVelm(0),
	m_dAcl(0),
	m_dOmegVel(0),
	m_dOmegAcl(0),
	m_dNewNeighborList(0),
	m_dCurrentNeighborList(0),
	m_dNumNeighbors(0),
	m_dtangwallfront(0),
	m_dtangwallback(0),
	m_dtangwallbottom(0),
	m_dtangwalltop(0),
	m_dtangwallleft(0),
	m_dtangwallright(0),
	m_dPurgatory(0),
	m_hPurgatory(0),
    m_gridSize(gridSize),
    m_timer(NULL),
    m_solverIterations(1),
	m_ntotal(0),
	stepCount(0),
	backupStepCount(1),
	smallParticleRadius(smallRadius),
	largeParticleRadius(largeRadius),
	m_gapThickness(gap),
	fallSteptime(0)
{
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    float3 worldSize = make_float3(0.0300f, 1.0f, 1.0f);
    float cellSize = 2.2f*largeParticleRadius;//1.1f*2.0f*0.002f;//smallParticleRadius;// 0.00125f;  // cell size equal to largest particle diameter
	maxDropHeight = 0.02f;
	m_feedSpacing = make_uint3(8,72,64); //size 2:1
//	m_feedSpacing = make_uint3(16,16,128); //size 1:1, diameter 2mm
//	m_feedSpacing = make_uint3(8,16,75); //size 1:1, density diameter 3mm

	m_hRadiiTemp = new float[m_feedSpacing.x*m_feedSpacing.y*m_feedSpacing.z];
	memset(m_hRadiiTemp,0,sizeof(float)*m_feedSpacing.x*m_feedSpacing.y*m_feedSpacing.z);

    // Set simulation parameters
    m_params.worldOrigin = make_float3(0.0f, 0.0f, 0.0f);
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
    m_params.gridSize = make_uint3(m_gridSize.x,m_gridSize.y,m_gridSize.z);
    m_params.numCells = m_numGridCells;
    m_params.numBodies = m_numParticles;
    m_params.particleRadius = largeParticleRadius; // Make this the "nominal" particle radius of the larger particle
	m_params.rmax = m_params.particleRadius * 2.5f; // Set the maximum cutoff radius for use in neighborlist updating
	m_params.maxNeighbors = NUM_CONTACTS;
	m_params.sizeratio = powf(largeParticleRadius/smallParticleRadius,3.0f); // Volume ratio of large to small particles
    m_params.worldOrigin = make_float3(0.0f, 0.0f, 0.0f);

    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
	
    m_params.nspring = 0.5f;
	m_params.tspring = 2.0f / 7.0f * m_params.nspring;
    m_params.ndamping = 0.2f;
    m_params.tdamping = 0.2f;
    m_params.cshear = 0.1f;
	m_params.bshear = 0.6f;
	m_params.coloumb = 0.5f;
	m_params.bcoloumb = 0.5f;
    m_params.attraction = 0.0f;
	m_params.bspring = 1.5f;
    m_params.bdamping = 0.2f;
    m_params.gravity = make_float3(0.0f, 0.0f, 0.0f);
    m_params.globalDamping = 1.0f;
	m_params.gapThickness = m_gapThickness;
	m_params.height = 0.42f;
	m_params.inletRatio = 1.0f;//1.0f;

	initVelocity = make_float3(0,-0.5f,0);
	maxFallTime = sqrtf(powf((initVelocity.y/(-m_params.gravity.y)),2.0f)+(2.0f/(-m_params.gravity.y)*maxDropHeight))+(initVelocity.y/(-m_params.gravity.y));
	printf("Max fall time %f\n",maxFallTime);

	std::cout << "Initializing particle system..." << std::endl;

    _initialize((int)m_numParticles);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
    const int ncolors = 4;
    int i = 0;
    float c[ncolors][3] =
    {
        { 0.8, 0.0, 0.8, },
        { 0.0, 0.298, 0.6, },
        { 0.0, 0.6, 0.298, },
        { 0.8, 0.8, 0.0, }
    };
    if(t >= 0.0005f && t < 0.00075f)
    {
    	t = (t-0.0005f)/(0.00025f);
    	i = 0;
    }
    else if(t >= 0.00075f && t < 0.0015f)
    {
    	t = (t-0.00075f)/(0.00075f);
    	i = 1;
    }
    else if(t >= 0.0015f && t < 0.0045f)
    {
    	t = (t-0.0015f)/(0.003f);
    	i = 2;
    }
    else if(t >= 0.0045f && t < 0.006f)
    {
    	t = (t-0.0045f)/0.0015f;
    	i = 3;
    }
    else
    {
    	t = 1.0f;
    }

    r[0] = lerp(c[i][0], c[i+1][0], t);
    r[1] = lerp(c[i][1], c[i+1][1], t);
    r[2] = lerp(c[i][2], c[i+1][2], t);
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;
	printf("Number of particles: %i\n",m_numParticles);

    // allocate host storage

    m_hPos = new float[m_numParticles*4];
    m_hVel = new float[m_numParticles*4];
	m_hAcl = new float[m_numParticles*4];
	m_hCol = new float[m_numParticles*4];
	m_hVelm = new float[m_numParticles*4];
	m_hMass = new float[m_numParticles];
	m_hOmegVel = new float[m_numParticles*4];
	m_hOmegVelm = new float[m_numParticles*4];
	m_hOmegAcl = new float[m_numParticles*4];
	m_hCurrentNeighborList = new float[m_numParticles*4*NUM_CONTACTS];
	m_hNewNeighborList = new float[m_numParticles*4*NUM_CONTACTS];
	m_hNumNeighbors = new uint[m_numParticles];
	m_htangwallfront = new float[m_numParticles*4];
	m_htangwallback = new float[m_numParticles*4];
	m_htangwallbottom = new float[m_numParticles*4];
	m_htangwalltop = new float[m_numParticles*4];
	m_htangwallleft = new float[m_numParticles*4];
	m_htangwallright = new float[m_numParticles*4];
	m_hTotalNormalForce = new float[1];
	m_hPurgatory = new uint[m_numParticles];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));
	memset(m_hAcl, 0, m_numParticles*4*sizeof(float));
	memset(m_hCol, 0, m_numParticles*4*sizeof(float));
	memset(m_hMass, 0, m_numParticles*sizeof(float));
	memset(m_hVelm, 0, m_numParticles*4*sizeof(float));
	memset(m_hOmegVel, 0, m_numParticles*4*sizeof(float));
	memset(m_hOmegVelm, 0, m_numParticles*4*sizeof(float));
	memset(m_hOmegAcl, 0, m_numParticles*4*sizeof(float));
	memset(m_hCurrentNeighborList, 0, m_numParticles*NUM_CONTACTS*4*sizeof(float));
	memset(m_hNewNeighborList, 0, m_numParticles*NUM_CONTACTS*4*sizeof(float));
	memset(m_hNumNeighbors, 0, m_numParticles*sizeof(uint));
	memset(m_htangwallfront, 0, m_numParticles*4*sizeof(float));
	memset(m_htangwallback, 0, m_numParticles*4*sizeof(float));
	memset(m_htangwallbottom, 0, m_numParticles*4*sizeof(float));
	memset(m_htangwalltop, 0, m_numParticles*4*sizeof(float));
	memset(m_htangwallleft, 0, m_numParticles*4*sizeof(float));
	memset(m_htangwallright, 0, m_numParticles*4*sizeof(float));
	memset(m_hTotalNormalForce, 0, sizeof(float));
	memset(m_hPurgatory, 0, m_numParticles*sizeof(uint));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

	m_hListHeads = new uint[m_numGridCells];
	memset(m_hListHeads, 0, m_numGridCells*sizeof(uint));

	m_hNextPointers = new uint[2*m_numParticles];
	memset(m_hNextPointers, 0, m_numParticles*2*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    if (m_bUseOpenGL)
    {
        m_posVbo = createVBO(memSize);
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
		m_velVbo = createVBO(memSize);
		registerGLBufferObject(m_velVbo, &m_cuda_velvbo_resource);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize));
		checkCudaErrors(cudaMalloc((void **)&m_cudaVelVBO, memSize));
    }

    //allocateArray((void **)&m_dVel, memSize);
	allocateArray((void **)&m_dMass, sizeof(float) * m_numParticles);
	allocateArray((void **)&m_dAcl, memSize);
	allocateArray((void **)&m_dVelm, memSize);
	allocateArray((void **)&m_dOmegVel, memSize);
	allocateArray((void **)&m_dOmegAcl, memSize);
	allocateArray((void **)&m_dOmegVelm, memSize);
	allocateArray((void **)&m_dTotalNormalForce, sizeof(float));

	size_t pitch_current;

	allocate2dArray((void **)&m_dCurrentNeighborList, &pitch_current, NUM_CONTACTS*4*sizeof(float), m_numParticles);  // At most 16 contacting neighbors. Might need to change this if not physically realistic.

	m_params.pitchCurrent = pitch_current;

	size_t pitch_new;

	allocate2dArray((void **)&m_dNewNeighborList, &pitch_new, NUM_CONTACTS*4*sizeof(float), m_numParticles);  // Duplicate of current neighbor list for use in neighbor list updating.

	m_params.pitchNew = pitch_new;

	allocateArray((void **)&m_dtangwallfront, memSize);
	allocateArray((void **)&m_dtangwallback, memSize);
	allocateArray((void **)&m_dtangwallbottom, memSize);
	allocateArray((void **)&m_dtangwalltop, memSize);
	allocateArray((void **)&m_dtangwallleft, memSize);
	allocateArray((void **)&m_dtangwallright, memSize);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);
	allocateArray((void **)&m_dSortedMass, sizeof(float) * m_numParticles);
	allocateArray((void **)&m_dSortedVelm, memSize);
	allocateArray((void **)&m_dSortedOmegVel, memSize);
	allocateArray((void **)&m_dSortedOmegVelm, memSize);
	allocateArray((void **)&m_dSortedNumNeighbors, sizeof(uint)*m_numParticles);
	allocateArray((void **)&m_dPurgatory, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));
	allocateArray((void **)&m_dNumNeighbors, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));
    copyArrayToDevice(m_dTotalNormalForce,m_hTotalNormalForce,0,sizeof(float));

    checkCudaErrors(cudaMemset(m_dPurgatory, 0xffffffff, m_numParticles*sizeof(uint)));

	allocateArray((void **)&m_dListHeads, m_numGridCells*sizeof(uint));
	allocateArray((void **)&m_dNextPointers, m_numParticles*2*sizeof(uint));

#if 1

    if (m_bUseOpenGL)
    {
        m_colorVBO = createVBO(m_numParticles*4*sizeof(float));
        registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        //glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
        //float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        //float *ptr = data;

        //for (uint i=0; i<m_numParticles; i++)
        //{
        //    float t = i / (float) m_numParticles;
#if 0
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
#else
         //   colorRamp(t, ptr);
         //   ptr+=3;
#endif
          //  *ptr++ = 1.0f;
        //}

        //glUnmapBufferARB(GL_ARRAY_BUFFER);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float)*numParticles*4));
    }

#endif

    sdkCreateTimer(&m_timer);

    setParameters(&m_params);

	if (m_bRestart == true)
	{
		openRestartFile(m_readPos);
	}

    m_bInitialized = true;

    printf("Initialization success!\n\n");

}

void
ParticleSystem::_finalize()
{
	printf("Destructor called");

    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
	delete [] m_hAcl;
	delete [] m_hMass;
	delete [] m_hVelm;
	delete [] m_hCol;
	delete [] m_hOmegVel;
	delete [] m_hOmegVelm;
	delete [] m_hOmegAcl;
	delete [] m_hCurrentNeighborList;
	delete [] m_hNewNeighborList;
	delete [] m_hNumNeighbors;
	delete [] m_htangwallfront;
	delete [] m_htangwallback;
	delete [] m_htangwallbottom;
	delete [] m_htangwalltop;
	delete [] m_htangwallleft;
	delete [] m_htangwallright;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;
	delete [] m_hListHeads;
	delete [] m_hNextPointers;
	delete [] m_hPurgatory;
	delete [] m_hRadiiTemp;
	delete [] m_hTotalNormalForce;

    //freeArray(m_dVel);
	freeArray(m_dMass);
	freeArray(m_dVelm);
	freeArray(m_dOmegVel);
	freeArray(m_dOmegAcl);
	freeArray(m_dOmegVelm);
	freeArray(m_dAcl);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);
	freeArray(m_dSortedOmegVel);
	freeArray(m_dSortedMass);
	freeArray(m_dSortedVelm);
	freeArray(m_dSortedOmegVelm);
	freeArray(m_dSortedNumNeighbors);
	freeArray(m_dCurrentNeighborList);
	freeArray(m_dNewNeighborList);
	freeArray(m_dtangwallfront);
	freeArray(m_dtangwallback);
	freeArray(m_dtangwallbottom);
	freeArray(m_dtangwalltop);
	freeArray(m_dtangwallleft);
	freeArray(m_dtangwallright);
	freeArray(m_dTotalNormalForce);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
	freeArray(m_dNumNeighbors);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);
	freeArray(m_dListHeads);
	freeArray(m_dNextPointers);
	freeArray(m_dPurgatory);

    if (m_bUseOpenGL)
    {
        unregisterGLBufferObject(m_cuda_posvbo_resource);
		unregisterGLBufferObject(m_cuda_velvbo_resource);
		unregisterGLBufferObject(m_cuda_colorvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
		glDeleteBuffers(1, (const GLuint *)&m_velVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
    }
    else
    {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
		checkCudaErrors(cudaFree(m_cudaVelVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
    }
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

	if(m_bOutput == 1 && stepCount % m_recordStep == 0 && stepCount > 2.0*ASD)			// File output
	{
		write2Bin();
		printf("Step count %i\n",stepCount);
		printf("Height %f\n",topWallPos); //print total normal force, to check what it is
//		printf("Total Normal Force d %f\n",*m_dTotalNormalForce); //print total normal force,to check what it is
		printf("Total Normal Force h %f\n",*m_hTotalNormalForce); //print total normal force,to check what it is
	}

	//	printf("Total Normal Force %f\n",*m_hTotalNormalForce); //print total normal force,to check what it is
		//top wall HERE!
		if (stepCount>0.2*ASD)
		{
			float topWallAclLast = topWallAcl;
			float topWallVelLast = topWallVel;
			topWallAcl = -*m_hTotalNormalForce/topWallMass - 9.81f; //negative of TNF, to stay consistent with up being positive
				if (topWallPos> 0.0) //145changediameter
				{
					topWallVel += topWallAcl*deltaTime;
					topWallPos += topWallVel*deltaTime;
				}

//		topWallPos += (topWallVel*deltaTime + 0.5f*topWallAcl*deltaTime*deltaTime);
//		topWallAcl = -*m_hTotalNormalForce/topWallMass - 9.81f; //negative of TNF, to stay consistent with up being positive
//		topWallVel += 0.5f*(topWallAclLast+topWallAcl)*deltaTime;
		}

	//	if (m_bOutput == 1 && stepCount>50000 && m_hTotalNormalForce[0] > prescribedTnf*0.97f)
	//	{
	//		write2Bin();
	//		printf("Total Normal Force %f\n",*m_hTotalNormalForce); //print total normal force,to check what it is
	//	}
	//
	//	if (m_bOutput == 1 && stepCount>50000 && m_hTotalNormalForce[0] < prescribedTnf*1.03f)
	//	{
	//		write2Bin();
	//		printf("Total Normal Force %f\n",*m_hTotalNormalForce); //print total normal force,to check what it is
	//	}
	m_hTotalNormalForce[0] = 0.0f; //resets total normal force on each step
	cudaMemcpy(m_dTotalNormalForce, m_hTotalNormalForce, sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(h_totalNormalForce, totalNormalForce, sizeof(float), cudaMemcpyDeviceToHost);
//	copyArrayToDevice(m_dTotalNormalForce,m_hTotalNormalForce,0,sizeof(float));

    float *dPos;
	float *dVel;

    if (m_bUseOpenGL)
    {
        dPos = (float *) mapGLBufferObject(&m_cuda_posvbo_resource);
		dVel = (float *) mapGLBufferObject(&m_cuda_velvbo_resource);
    }
    else
    {
        dPos = (float *) m_cudaPosVBO;
		dVel = (float *) m_cudaVelVBO;
    }


    // update constants
    setParameters(&m_params);

    // integrate
    integrateSystem(
        dPos,
        dVel,
		m_dAcl,
		m_dVelm,
		m_dOmegVel,
		m_dOmegAcl,
		m_dOmegVelm,
        deltaTime,
        m_ntotal);

	periodicBoundaries(dPos, 0.2816f, 0.0352f, m_ntotal);
//    periodicBoundaries(dPos, 0.1408f, 0.0176f, m_ntotal);  //changediameter
    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_ntotal);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_ntotal);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dSortedVel,
		m_dSortedMass,
		m_dSortedVelm,
		m_dSortedOmegVel,
		m_dSortedOmegVelm,
		m_dSortedNumNeighbors,
        m_dGridParticleHash,
        m_dGridParticleIndex,
		m_dNumNeighbors,
        dPos,
        dVel,
		m_dMass,
		m_dVelm,
		m_dOmegVel,
		m_dOmegVelm,
        m_ntotal,
        m_numGridCells);

	computeNeighborList(m_dNewNeighborList,
						m_params.pitchNew,
						m_dCurrentNeighborList,
						m_params.pitchCurrent,
						m_dNumNeighbors,
						m_dSortedPos,
						m_dCellStart,
						m_dCellEnd,
						m_dGridParticleIndex,
						m_ntotal,
						m_numGridCells,
						m_params.maxNeighbors);

    // process collisions
    collide(
        m_dAcl,
		m_dOmegAcl,
		m_dSortedPos,
		dPos,
        m_dSortedVel,
		dVel,
		m_dSortedMass,
		m_dMass,
		m_dSortedVelm,
		m_dSortedOmegVel,
		m_dOmegVel,
		m_dSortedOmegVelm,
        m_dGridParticleIndex,
		m_dNewNeighborList,
		m_dNumNeighbors,
		m_dtangwallfront,
		m_dtangwallback,
		m_dtangwallleft,
		m_dtangwallright,
		m_dtangwallbottom,
		m_dtangwalltop,
		m_dPurgatory,
        m_dCellStart,
        m_dCellEnd,
        m_ntotal,
        m_numGridCells,
		deltaTime,
		stepCount,
		m_dTotalNormalForce,
		m_hTotalNormalForce,
		topWallPos,
		topWallMass,
		shearRate);

//    copyArrayFromDevice(m_hTotalNormalForce,m_dTotalNormalForce,0,sizeof(float));

    // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (m_bUseOpenGL)
    {
        unmapGLBufferObject(m_cuda_posvbo_resource);
		unmapGLBufferObject(m_cuda_velvbo_resource);
    }

	stepCount++;

}

void
ParticleSystem::dumpNeighborLists()
{
	copyArrayFromDevice(m_hListHeads, m_dListHeads, 0, sizeof(uint)*m_numGridCells);
	printf("List heads:\n");
	for (uint i = 0; i<m_numGridCells; i++)
	{
		printf("%i\n", m_hListHeads[i]);
	}
}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
    uint maxCellSize = 0;

    for (uint i=0; i<m_numGridCells; i++)
    {
		printf("%i\n", m_hCellStart[i]);
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            //            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float)*4*count);
    copyArrayFromDevice(m_hVel, 0, &m_cuda_velvbo_resource, sizeof(float)*4*count);
//	copyArrayFromDevice(m_hOmegVel, m_dOmegVel, 0, sizeof(float)*4*count);
//	copyArrayFromDevice(m_hVelMag, m_dVelMag, 0, sizeof(float)*count);
//	copyArrayFromDevice(m_hMass, m_dMass, 0, sizeof(float)*count);
	cudaMemcpy(m_hNumNeighbors, m_dNumNeighbors, sizeof(uint)*count, cudaMemcpyDeviceToHost);
//	copyArrayFromDevice(m_htangwallfront, m_dtangwallfront, 0, sizeof(float)*4*count);
//	copyArrayFromDevice(m_htangwallback, m_dtangwallback, 0, sizeof(float)*4*count);
//	copyArrayFromDevice(m_htangwallleft, m_dtangwallleft, 0, sizeof(float)*4*count);
//	copyArrayFromDevice(m_htangwallright, m_dtangwallright, 0, sizeof(float)*4*count);

//    for (uint i=500; i<501; i++)
//    {
//        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
//        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
////		printf("vel mag: (%2.4f)\n", m_hVelMag[i]);
//		printf("ang vel: (%.8f, %.8f, %.8f, %.8f)\n", m_hOmegVel[i*4+0], m_hOmegVel[i*4+1], m_hOmegVel[i*4+2], m_hOmegVel[i*4+3]);
////		printf("tangdisp - front: (%.6f, %.6f, %.6f, %1.2f)\n", m_htangwallfront[i*4+0], m_htangwallfront[i*4+1], m_htangwallfront[i*4+2], m_htangwallfront[i*4+3]);
////		printf("tangdisp - back: (%.6f, %.6f, %.6f, %1.2f)\n", m_htangwallback[i*4+0], m_htangwallback[i*4+1], m_htangwallback[i*4+2], m_htangwallback[i*4+3]);
////		printf("tangdisp - left: (%.6f, %.6f, %.6f, %1.2f)\n", m_htangwallleft[i*4+0], m_htangwallleft[i*4+1], m_htangwallleft[i*4+2], m_htangwallleft[i*4+3]);
////		printf("tangdisp - right: (%.6f, %.6f, %.6f, %1.2f)\n", m_htangwallright[i*4+0], m_htangwallright[i*4+1], m_htangwallright[i*4+2], m_htangwallright[i*4+3]);
//		printf("mass: (%.8f)\n", m_hMass[i]);
//    }
	for (uint i=0;i<2;i++)
	{
		printf("%i\n",m_hNumNeighbors[i]);
	}

    for (uint i=0; i<2; i++)
    {
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
		printf("ang vel: (%.8f, %.8f, %.8f, %.8f)\n", m_hOmegVel[i*4+0], m_hOmegVel[i*4+1], m_hOmegVel[i*4+2], m_hOmegVel[i*4+3]);
		printf("tangdisp - front: (%.6f, %.6f, %.6f, %1.2f)\n", m_htangwallfront[i*4+0], m_htangwallfront[i*4+1], m_htangwallfront[i*4+2], m_htangwallfront[i*4+3]);
		printf("tangdisp - back: (%.6f, %.6f, %.6f, %1.2f)\n", m_htangwallback[i*4+0], m_htangwallback[i*4+1], m_htangwallback[i*4+2], m_htangwallback[i*4+3]);
		printf("tangdisp - left: (%.6f, %.6f, %.6f, %1.2f)\n", m_htangwallleft[i*4+0], m_htangwallleft[i*4+1], m_htangwallleft[i*4+2], m_htangwallleft[i*4+3]);
		printf("tangdisp - right: (%.6f, %.6f, %.6f, %1.2f)\n", m_htangwallright[i*4+0], m_htangwallright[i*4+1], m_htangwallright[i*4+2], m_htangwallright[i*4+3]);
		printf("mass: (%.8f)\n", m_hMass[i]);
    }
	printContacts();
}

float *
ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    switch (array)
    {
        default:
        case POSITION:
            hdata = m_hPos;
        	ddata = m_cudaPosVBO;
            if (m_bUseOpenGL)
            {
                ddata = m_dPos;
            	cuda_vbo_resource = m_cuda_posvbo_resource;
            }
            break;

        case VELOCITY:
            hdata = m_hVel;
        	ddata = m_cudaVelVBO;
            if (m_bUseOpenGL)
            {
                ddata = m_dVel;
            	cuda_vbo_resource = m_cuda_velvbo_resource;
            }
            break;

		case CONTACTS:
			hdata = m_hCurrentNeighborList;
			ddata = m_dCurrentNeighborList;
			copy2dArrayFromDevice(hdata, NUM_CONTACTS*4*sizeof(float), ddata, m_params.pitchCurrent, NUM_CONTACTS*4*sizeof(float), m_numParticles);
			return hdata;
    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
        default:
        case POSITION:
            {
                if (m_bUseOpenGL)
                {
                    unregisterGLBufferObject(m_cuda_posvbo_resource);
                    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
                    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                    registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
                }
                else
                {
                	copyArrayToDevice(m_cudaPosVBO,data,start*4*sizeof(float),count*4*sizeof(float));
                }
            }
            break;

        case VELOCITY:
			{
				if (m_bUseOpenGL)
				{
					unregisterGLBufferObject(m_cuda_velvbo_resource);
					glBindBuffer(GL_ARRAY_BUFFER, m_velVbo);
					glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
					glBindBuffer(GL_ARRAY_BUFFER, 0);
					registerGLBufferObject(m_velVbo, &m_cuda_velvbo_resource);
				}
				else
				{
		            copyArrayToDevice(m_cudaVelVBO, data, start*4*sizeof(float), count*4*sizeof(float));
				}
			}
            break;

		case ACCELERATION:
			copyArrayToDevice(m_dAcl, data, start*4*sizeof(float), count*4*sizeof(float));
			break;

		case MASS:
			copyArrayToDevice(m_dMass, data, start*sizeof(float), count*sizeof(float));
			break;

		case VELOCITYM:
			copyArrayToDevice(m_dVelm, data, start*4*sizeof(float), count*4*sizeof(float));
			break;

		case ANGULARVEL:
			copyArrayToDevice(m_dOmegVel, data, start*4*sizeof(float), count*4*sizeof(float));
			break;

		case ANGULARVELM:
			copyArrayToDevice(m_dOmegVelm, data, start*4*sizeof(float), count*4*sizeof(float));
			break;

		case ANGULARACL:
			copyArrayToDevice(m_dOmegAcl, data, start*4*sizeof(float), count*4*sizeof(float));
			break;

		case COLOR:
			{
				if (m_bUseOpenGL)
				{
					unregisterGLBufferObject(m_cuda_colorvbo_resource);
					glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
					glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
					glBindBuffer(GL_ARRAY_BUFFER, 0);
					registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);
				}
			}
			break;
    }
}

void
ParticleSystem::openRestartFile(int readPos)
{
	std::cout << "Opening binary file and seeking position after the last recorded frame...";
	m_restartFile = fopen(outputFileName.c_str(), "r+b");
	if(m_restartFile == NULL)
	{
		std::cerr << "Error opening binary file." << std::endl;
	}
	int err = fseek(m_restartFile, readPos, SEEK_SET);
	if (err != 0)
	{
		std::cerr << "Error seeking to last recorded frame.";
	}
	else
	{
		std::cout << "Success.\n\nResuming simulation..." << std::endl;
	}
}
void
ParticleSystem::write2Bin()
{
	float *pos;
	float *vel;
	float *heightp = new float[1];
	float *tot = new float[1];
	struct cudaGraphicsResource *cuda_vbo_resource = 0;

	heightp[0] = topWallPos;

	if (m_bRestart == true)
	{
		goto write;
	}

	FILE * pFile;
	pFile = fopen("output.bin", "a+b");
	//pFile = fopen(outputFileName.c_str(), "w+");

	pos = getArray(POSITION);
	vel = getArray(VELOCITY);
	copyArrayFromDevice(m_hOmegVel, m_dOmegVel, &cuda_vbo_resource, sizeof(float)*4*m_numParticles);
	tot[0] = (float)m_ntotal;
	fwrite(tot, sizeof(float), 1, pFile);
	fwrite(heightp, sizeof(float), 1, pFile);
	for (int i = 0; i < m_ntotal; i++)
	{
		float buffer[12] = {pos[i*4+3],(float)i,vel[i*4+3],pos[i*4+0],pos[i*4+1],pos[i*4+2],vel[i*4+0],vel[i*4+1],vel[i*4+2],m_hOmegVel[i*4+0],m_hOmegVel[i*4+1],m_hOmegVel[i*4+2]}; // radius, index, color, x, y, z, vx, vy, vz, ovx, ovy, ovz
		fwrite(buffer, sizeof(float), 12, pFile);
		//fprintf(pFile, "%f %f %f %f %f %f %f %f %f \n", pos[i*4+3],(float)i,vel[i*4+3],pos[i*4+0],pos[i*4+1],pos[i*4+2],vel[i*4+0],vel[i*4+1],vel[i*4+2]);
	}
	fclose(pFile);
	return;

write:
	pos = getArray(POSITION);
	vel = getArray(VELOCITY);
	tot[0] = (float)m_ntotal;
	fwrite(tot, sizeof(float), 1, m_restartFile);
	for (int i = 0; i < m_ntotal; i++)
	{
		float buffer[9] = {pos[i*4+3],(float)i,vel[i*4+3],pos[i*4+0],pos[i*4+1],pos[i*4+2],vel[i*4+0],vel[i*4+1],vel[i*4+2]}; // radius, index, color, x, y, z, vx, vy, vz
		fwrite(buffer, sizeof(float), 9, m_restartFile);
	}
	return;
}

void
ParticleSystem::printContacts()
{
	copy2dArrayFromDevice(m_hCurrentNeighborList, NUM_CONTACTS*4*sizeof(float), m_dCurrentNeighborList, m_params.pitchCurrent, NUM_CONTACTS*4*sizeof(float), m_ntotal);
	for (uint i = 0; i < m_ntotal; i++) {
		for (int j = 0; j < NUM_CONTACTS; j++) {
			char line[192];
			sprintf(line, "%1.0f, %1.12f, %1.12f, %1.12f | ", m_hCurrentNeighborList[i*NUM_CONTACTS*4 + j*4], m_hCurrentNeighborList[i*NUM_CONTACTS*4 + j*4 + 1], m_hCurrentNeighborList[i*NUM_CONTACTS*4 + j*4 + 2], m_hCurrentNeighborList[i*NUM_CONTACTS*4 + j*4 + 3]);
			std::cout << line;
		}
		std::cout << "\n";
	}
}

void
ParticleSystem::initContactList()
{
	size_t spitch = NUM_CONTACTS*4*sizeof(float);
	for (uint i = 0; i < m_numParticles; i++) {
		for (int j = 0; j < NUM_CONTACTS; j++) {

			m_hCurrentNeighborList[i*NUM_CONTACTS*4 + j*4] = 0.0f;
			m_hCurrentNeighborList[i*NUM_CONTACTS*4 + j*4 + 1] = 0.0f;
			m_hCurrentNeighborList[i*NUM_CONTACTS*4 + j*4 + 2] = 0.0f;
			m_hCurrentNeighborList[i*NUM_CONTACTS*4 + j*4 + 3] = 0.0f;

			m_hNewNeighborList[i*NUM_CONTACTS*4 + j*4] = 0.0f;
			m_hNewNeighborList[i*NUM_CONTACTS*4 + j*4 + 1] = 0.0f;
			m_hNewNeighborList[i*NUM_CONTACTS*4 + j*4 + 2] = 0.0f;
			m_hNewNeighborList[i*NUM_CONTACTS*4 + j*4 + 3] = 0.0f;

		}
		m_hNumNeighbors[i] = NUM_CONTACTS;
	}
	
	copy2dArrayToDevice(m_dCurrentNeighborList, m_params.pitchCurrent, m_hCurrentNeighborList, spitch, NUM_CONTACTS*4*sizeof(float), m_numParticles);
	copy2dArrayToDevice(m_dNewNeighborList, m_params.pitchNew, m_hNewNeighborList, spitch, NUM_CONTACTS*4*sizeof(float), m_numParticles);
	copyArrayToDevice(m_dNumNeighbors, m_hNumNeighbors, 0, m_numParticles*sizeof(uint));
}

void
ParticleSystem::initBCList()
{
	copyArrayToDevice(m_dtangwallfront, m_htangwallfront, 0, m_numParticles*4*sizeof(float));
	copyArrayToDevice(m_dtangwallback, m_htangwallback, 0, m_numParticles*4*sizeof(float));
	copyArrayToDevice(m_dtangwallbottom, m_htangwallbottom, 0, m_numParticles*4*sizeof(float));
	copyArrayToDevice(m_dtangwalltop, m_htangwalltop, 0, m_numParticles*4*sizeof(float));
	copyArrayToDevice(m_dtangwallleft, m_htangwallleft, 0, m_numParticles*4*sizeof(float));
	copyArrayToDevice(m_dtangwallright, m_htangwallright, 0, m_numParticles*4*sizeof(float));
}

void
ParticleSystem::reset(ParticleConfig config)
{
    switch (config)
    {
        default:
        case CONFIG_RANDOM:
            {
                int p = 0, v = 0;

                for (uint i=0; i < m_numParticles; i++)
                {
                    float point[3];
                    point[0] = frand();
                    point[1] = frand();
                    point[2] = frand();
                    m_hPos[p++] = 2 * (point[0] - 0.5f);
                    m_hPos[p++] = 2 * (point[1] - 0.5f);
                    m_hPos[p++] = 2 * (point[2] - 0.5f);
                    m_hPos[p++] = 1.0f; // radius
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                }
            }
            break;

        case CONFIG_GRID:
            {
                initContactList();
				initBCList();
				addParticles();
            }
            break;
    }

}

void
ParticleSystem::addParticles()
{

	std::cout << "addParticles" << std::endl;
	int i = 0;

//	 First particle

		for (int y = 0; y < m_feedSpacing.y; y++)
		{
			for (int z = 0; z < m_feedSpacing.z; z++)
			{
				for (int x = 0; x < m_feedSpacing.x; x++)
				{
					//srand(time(NULL));
					float h = frand();

//					m_hPos[i * 4] = 0.0005f+ x * 0.001f + (2.0*h-1.0)*0.0001f;
//					m_hPos[i * 4 + 1] = 0.0005f + y*0.001f;
//					m_hPos[i * 4 + 2] = 0.0005f + z * 0.001f + (2.0*h - 1.0)*0.0001f;

					//size bi
					m_hPos[i*4] = (2*largeParticleRadius * x*1.1f) + largeParticleRadius*1.1f+(frand()*2.0f-1.0f)*largeParticleRadius*0.1f;
					if (y<4)
					{
						m_hPos[i*4+1] = (2*largeParticleRadius * y*1.1f) + 2*largeParticleRadius*1.1f;
					}
					else
					{
						m_hPos[i*4+1] = (2*largeParticleRadius * 4*1.1f) + 2*largeParticleRadius*1.1f*(y-2);
					}

					m_hPos[i*4+2] = (2*largeParticleRadius * z*1.1f) + largeParticleRadius*1.1f+(frand()*2.0f-1.0f)*largeParticleRadius*0.1f;
					m_hPos[i*4+3] = smallParticleRadius;

//					//density bi
//					m_hPos[i*4] = 0.0352f/9.0f*x+0.0352f/9.0f+(frand()*2.0f-1.0f)*largeParticleRadius*0.1f;// (2*largeParticleRadius * x*1.2f) + largeParticleRadius*1.2f+(frand()*2.0f-1.0f)*largeParticleRadius*0.1f;
//					m_hPos[i*4+1] = (2*largeParticleRadius * y*1.2f) + largeParticleRadius*1.2f+(frand()*2.0f-1.0f)*0.0f;
//					m_hPos[i*4+2] = 0.2816f/76.0f*z+0.2816f/76.0f+(frand()*2.0f-1.0f)*largeParticleRadius*0.1f;//(2*largeParticleRadius * z*1.2f) + largeParticleRadius*1.2f+(frand()*2.0f-1.0f)*largeParticleRadius*0.1f;
//					m_hPos[i*4+3] = smallParticleRadius;

					m_hVelm[i * 4] = 0.0f;
					m_hVelm[i * 4 + 1] = 0.0f;
					m_hVelm[i * 4 + 2] = 0.0f;
					m_hVelm[i * 4 + 3] = 0.0f;

					m_hOmegVel[i * 4] = 0.0f;
					m_hOmegVel[i * 4 + 1] = 0.0f;
					m_hOmegVel[i * 4 + 2] = 0.0f;
					m_hOmegVel[i * 4 + 3] = 0.0f;

					m_hOmegAcl[i * 4] = 0.0f;
					m_hOmegAcl[i * 4 + 1] = 0.0f;
					m_hOmegAcl[i * 4 + 2] = 0.0f;
					m_hOmegAcl[i * 4 + 3] = 0.0f;

					m_hAcl[i * 4] = 0.0f;
					m_hAcl[i * 4 + 1] = 0.0f;
					m_hAcl[i * 4 + 2] = 0.0f;
					m_hAcl[i * 4 + 3] = 0.0f;

					m_hOmegVelm[i * 4] = 0.0f;
					m_hOmegVelm[i * 4 + 1] = 0.0f;
					m_hOmegVelm[i * 4 + 2] = 0.0f;
					m_hOmegVelm[i * 4 + 3] = 0.0f;

					m_hVel[i * 4] = 0.0f;
					m_hVel[i * 4 + 1] = 0.0f;
					m_hVel[i * 4 + 2] = 0.0f;
					m_hVel[i * 4 + 3] = 1.0f;

					m_hCol[i * 4] = 1.0f;
					m_hCol[i * 4 + 1] = 0.0f;
					m_hCol[i * 4 + 2] = 0.0f;
					m_hCol[i * 4 + 3] = 1.0f;

					m_hMass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(m_hPos[i * 4 + 3], 3) * 2500.0f;

					i++;

//					if (i >= m_numParticles)
//					{
//						goto set;
//					}
				}
			}
		}

//	Creates large particles in grid, at random locations, in order to make prescribed large/small particle ratio
		i = 0;
		float nratio = m_params.inletRatio * (1.0f / m_params.sizeratio);
		float ratio = nratio / (1 + nratio);
		do
		{
			float h = frand();
				if (m_hVel[int(m_numParticles * h)*4+3] == 1.0f) {
					m_hVel[int(m_numParticles * h)*4+3] = 2.0f;
					i++;
				}
//								if (m_hVel[(int(m_numParticles )-i-1)*4+3]==1.0f){
//									m_hVel[(int(m_numParticles )-i-1)*4+3]=2.0f;
//									i++;
//								}
		}
		while (i < int(m_numParticles * ratio));

	//	Create slight polydispersity and colors of particles
	//	random locations
		i = 0;
		do
		{
			float h =0.5f;// frand();
			if (m_hVel[i*4+3] == 1.0f){
//				m_hPos[i*4+3] = (2.0f*h - 1.0f) * 0.1f*smallParticleRadius + smallParticleRadius; //size bidisperse
//				m_hMass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(m_hPos[i*4+3],3) * 2500.0f; //size bi
				m_hPos[i*4+3] = (2.0f*h - 1.0f) * 0.2f*smallParticleRadius + smallParticleRadius; //density bidisperse
				m_hMass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(m_hPos[i*4+3],3) *  1.0f * RHO_L; //density bi
				m_hCol[i*4] = 1.0;
				m_hCol[i*4+1] = 0.2;   // Particle colors (RED)
				m_hCol[i*4+2] = 0.2;
				m_hCol[i*4+3] = 1.0;
//				m_hMass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(m_hPos[i*4+3],3) * 1.0f * 2500.0f;
			}
			else {
//				m_hPos[i*4+3] = (2.0f*h - 1.0f) * 0.1f*largeParticleRadius + largeParticleRadius; //size bidisperse
//				m_hMass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(m_hPos[i*4+3],3) * 2500.0f; //size bidisperse
				m_hPos[i*4+3] = (2.0f*h - 1.0f) * 0.2f*largeParticleRadius + largeParticleRadius; //Density bi
				m_hMass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(m_hPos[i*4+3],3) * 1.0f * RHO_H; //Density bi
				m_hCol[i*4] = 0.2;
				m_hCol[i*4+1] = 0.2;    // Particle colors (BLUE)
				m_hCol[i*4+2] = 1.0;
				m_hCol[i*4+3] = 1.0;
//				m_hMass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(m_hPos[i*4+3],3) * 1.0f * 2500.0f;

			}
			//m_hMass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(m_hPos[i*4+3],3) * 2500.0f;
			i++;
		}
		while (i < (int)m_numParticles);


//set:

	setArray(POSITION, m_hPos, 0, m_numParticles);
	setArray(ACCELERATION, m_hAcl, 0, m_numParticles);
	setArray(ANGULARVELM, m_hOmegVelm, 0, m_numParticles);
	setArray(VELOCITYM, m_hVelm, 0, m_numParticles);
	setArray(VELOCITY, m_hVel, 0, m_numParticles);
	setArray(ANGULARACL, m_hOmegAcl, 0, m_numParticles);
	setArray(MASS, m_hMass, 0, m_numParticles);
	setArray(ANGULARVEL, m_hOmegVel, 0, m_numParticles);
	setArray(COLOR, m_hCol, 0, m_numParticles);

	m_ntotal += m_numParticles;

}
