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

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#define USE_TEX 0

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"
typedef unsigned int uint;

// forces struct
struct Forces
{
	float3 force;
	float3 torque;
};

// simulation parameters
struct SimParams
{

    float3 gravity;
    float globalDamping;
    float particleRadius;
	float sizeratio;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;
	float rmax;

	float gapThickness;

    uint numBodies;
    uint maxParticlesPerCell;
	uint maxNeighbors;
	size_t pitchCurrent;
	size_t pitchNew;

	bool orifice;
	float height;

    float nspring;
	float tspring;
	//float poisson1;
	//float poisson2;
	//float young1;
	//float young2;
    float ndamping;
    float tdamping;
    float cshear;
	float bshear;
	float coloumb;
	float bcoloumb;
	float rollfc;
	float rollfb;
    float attraction;
	float bspring;
    float bdamping;
    float inletRatio;
};

#endif
