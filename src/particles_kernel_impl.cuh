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

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#define NUM_CONTACTS 32

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"
#include <thrust/device_vector.h>

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;
texture<float4, 1, cudaReadModeElementType> oldVelmTex;
texture<float4, 1, cudaReadModeElementType> oldOmegVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;



struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
		volatile float4 aclData = thrust::get<2>(t);
		volatile float4 velmData = thrust::get<3>(t);
		volatile float4 omegvelData = thrust::get<4>(t);
		volatile float4 omegaclData = thrust::get<5>(t);
		volatile float4 omegvelmData = thrust::get<6>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);
		float3 acl = make_float3(aclData.x, aclData.y, aclData.z);
		float3 velm = make_float3(velmData.x, velmData.y, velmData.z);
		float3 omegvel = make_float3(omegvelData.x, omegvelData.y, omegvelData.z);
		float3 omegacl = make_float3(omegaclData.x, omegaclData.y, omegaclData.z);
		float3 omegvelm = make_float3(omegvelmData.x, omegvelmData.y, omegvelData.z);


		//old algorithm
//		float3 tempvel = deltaTime * acl;
//		float3 tempomegvel = deltaTime * omegacl;
//
//		velm = vel + 0.5f * tempvel;
//		omegvelm = omegvel + 0.5f * tempomegvel;
//
//		pos += deltaTime * (vel + 0.5f * tempvel);
//
//		omegvel += 0.65f * tempomegvel;
//		vel += 0.65f * tempvel;
//
//        vel *= params.globalDamping;
//
//		acl.x = acl.y = acl.z = 0.0f;
//		omegacl.x = omegacl.y = omegacl.z = 0.0f;

		//new algorithm

		float3 tempvel = deltaTime * acl;
		float3 tempomegvel = deltaTime * omegacl;

		velm = vel + tempvel;
		omegvelm = omegvel + tempomegvel;
		pos += deltaTime * velm;
		vel = velm;
		omegvel = omegvelm;

		acl.x = acl.y = acl.z = 0.0f;
		omegacl.x = omegacl.y = omegacl.z = 0.0f;

//				 //this is the testing velocity updating algorithm
//
//		velm = vel + 0.5f * deltaTime * acl;
//		omegvelm = omegvel + 0.5f * deltaTime * omegacl;
//
//		pos += deltaTime * velm;
//
//		acl.x = acl.y = acl.z = 0.0f;
//		omegacl.x = omegacl.y = omegacl.z = 0.0f;


// store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
		thrust::get<2>(t) = make_float4(acl, aclData.w);
		thrust::get<3>(t) = make_float4(velm, velmData.w);
		thrust::get<4>(t) = make_float4(omegvel, omegvelData.w);
		thrust::get<5>(t) = make_float4(omegacl, omegaclData.w);
		thrust::get<6>(t) = make_float4(omegvelm, omegvelmData.w);
    }
};

struct periodic_functor
{
	float maxZ;
	float maxX;

	__host__ __device__
	periodic_functor(float maximumZ, float maximumX) : maxZ(maximumZ), maxX(maximumX) {}

	__device__
	void operator()(float4& Pos)
	{
		float4 temp = Pos;
		float posz = temp.z;
		float posy = temp.y;
		float posx = temp.x;

		if(posz >= maxZ)
		{
			posz -= maxZ;
		}
		if(posz < 0)
		{
			posz += maxZ;
		}
		if (posx >= maxX)
		{
			posx -= maxX;
		}
		if (posx < 0)
		{
			posx += maxX;
		}
		Pos = make_float4(posx,posy,posz,temp.w);
	}
};

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}


// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float4 *sortedPos,        // output: sorted positions
                                  float4 *sortedVel,        // output: sorted velocities
								  float *sortedMass,		// output: sorted masses
								  float4 *sortedVelm,       // output: sorted intermediate velocities
								  float4 *sortedOmegVel,    // output: sorted angular velocities
								  float4 *sortedOmegVelm,	// output: sorted intermediate angular velocities
								  uint   *sortedNumNeighbors, //output: sorted numNeighbors array
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
								  uint   *d_numNeighbors,   // input: numNeighbors array
                                  float4 *oldPos,           // input: position array
                                  float4 *oldVel,           // input: velocity array
								  float  *oldMass,			// input: masses array
								  float4 *oldVelm,          // input: intermediate velocity array
								  float4 *oldOmegVel,       // input: angular velocity array
								  float4 *oldOmegVelm,		// input: intermediate angular velocity array
                                  uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread

        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    __syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data

        uint sortedIndex = gridParticleIndex[index];

        float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
        float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh
		float mass = FETCH(oldMass, sortedIndex);
		float4 velm = FETCH(oldVelm, sortedIndex);
		float4 omegvel = FETCH(oldOmegVel, sortedIndex);
		float4 omegvelm = FETCH(oldOmegVelm, sortedIndex);
		uint numNeighbors = FETCH(d_numNeighbors, sortedIndex);

        sortedPos[index] = pos;
        sortedVel[index] = vel;
		sortedMass[index] = mass;
		sortedVelm[index] = velm;
		sortedOmegVel[index] = omegvel;
		sortedOmegVelm[index] = omegvelm;
		sortedNumNeighbors[index] = numNeighbors;
    }


}

// collide two spheres using DEM method
__device__
Forces collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
					  float massA, float massB,
					  float3 omegvelA, float3 omegvelB,
                      float radiusA, float radiusB,
                      float attraction, uint pid, uint nebIndex, float *newNeighborList,
					  float deltaTime)
{
    // calculate relative position
    float3 relPos = posA - posB;

	// recalculate the normalized position using the Minimum Image Convention
	while(relPos.z<-0.1408f) relPos.z+=0.2816f;
	while(relPos.z>=0.1408f) relPos.z-=0.2816f;
	while (relPos.x<-0.0176) relPos.x += 0.0352f;
	while (relPos.x >= 0.0176) relPos.x -= 0.0352f;

//	while (relPos.z<-0.0704) relPos.z += 0.1408f;
//	while (relPos.z >= 0.0704) relPos.z -= 0.1408f;
//	while (relPos.x<-0.0088) relPos.x += 0.0176f;
//	while (relPos.x >= 0.0088) relPos.x -= 0.0176f;


    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

	Forces forces;
	forces.force = make_float3(0.0f);
	forces.torque = make_float3(0.0f);
    float3 force_n = make_float3(0.0f);
	float3 force_s = make_float3(0.0f);
	float3 tangdisp = make_float3(0.0f);

    if (dist < collideDist)
    {
		// unit normal vector
        float3 norm = relPos / dist;

		// effective mass
		float masseff = massA * massB / (massA + massB);

        // relative velocity vector
        float3 relVel = velA - velB;
		
		// normal relative velocity vector
		float3 relVeln = dot(relVel,norm) * norm;

        // relative tangential velocity
        float3 tanVel = relVel - relVeln - cross((1.0f/(radiusA + radiusB)*(radiusA*omegvelA + radiusB*omegvelB)), relPos);

        // spring force
        force_n = params.nspring * masseff * (collideDist - dist) * norm;   // Normal elastic force

        // dashpot (damping) force
        force_n += -2.0f * params.ndamping * masseff * relVeln;   // Normal velocity-dependent damping

		float force_n_len = length(force_n);  // Magnitude of normal force

		if(!isnan(force_n_len)){forces.force = force_n;};

        // tangential ELASTIC shear force

		size_t rowIndex_new = pid*(params.pitchNew/sizeof(float));

		newNeighborList[rowIndex_new + nebIndex*4+1] += tanVel.x * deltaTime;
		newNeighborList[rowIndex_new + nebIndex*4+2] += tanVel.y * deltaTime;
		newNeighborList[rowIndex_new + nebIndex*4+3] += tanVel.z * deltaTime;
		
		tangdisp.x = newNeighborList[rowIndex_new + nebIndex*4+1];
		tangdisp.y = newNeighborList[rowIndex_new + nebIndex*4+2];
		tangdisp.z = newNeighborList[rowIndex_new + nebIndex*4+3];

		// Update the tangential displacement right here

		force_s = params.tspring * masseff * tangdisp;
		force_s += 2.0f * params.tdamping * masseff *tanVel;

		float force_s_len = length(force_s);
		if (force_s_len > 0) {
		force_s = force_s / force_s_len;
		}
		else {
		force_s.x = force_s.y = force_s.z = 0.0f;
		}
		// Crude implementation of sign function
		//float3 tansgn=make_float3((tangdisp.x>0)?1.0f:(((tangdisp.x)<0)?-1.0f:0.0f),(tangdisp.y>0)?1.0f:(((tangdisp.y)<0)?-1.0f:0.0f),(tangdisp.z>0)?1.0f:(((tangdisp.z)<0)?-1.0f:0.0f));

		force_s = -1.0f*fminf(force_s_len, params.coloumb * force_n_len) * force_s;   // Coloumb static friction law

//		if(!isnan(force_n_len)){
			forces.force += force_s;
			// torque
			forces.torque += radiusA * cross(norm, force_s);
//		}

		// rolling friction
		float omegvel_len = length(omegvelA);
		if (omegvel_len > 0)
		{
			forces.torque += params.rollfc * force_n_len * omegvelA / omegvel_len;
		}

        // attraction force
        forces.force += attraction*relPos;
        
     }
        
     return forces;
}

__device__
Forces collide2dfront(uint index,
					  float3 posA,
                      float3 velA,
					  float massA,
                      float radiusA,
					  float3 omegaA,
                      float attraction,
					  float *tangwall,
					  float deltaTime)
{
    // calculate relative position
    float3 Pos = posA;

    float dist = Pos.x;
    float collideDist = params.gapThickness - radiusA;

	Forces forces;
	forces.force = make_float3(0.0f);
	forces.torque = make_float3(0.0f);
	float3 force_n = make_float3(0.0f);
	float3 force_s = make_float3(0.0f);
	float3 tangdisp = make_float3(0.0f);

    if (dist > collideDist)
    {
        float3 norm;
		norm.x = -1.0f;
		norm.y = 0;
		norm.z = 0;

        // relative velocity
        float3 relVel = velA;

		// relative normal velocity
		float3 relVeln = dot(relVel,norm) * norm;

        // relative tangential velocity
        float3 tanVel = relVel - relVeln - radiusA * cross(omegaA, norm);

        // spring force
        force_n = params.bspring * massA * (dist - collideDist) * norm;

        // dashpot (damping) force
        force_n += -2.0f*params.bdamping * massA * relVeln;

		float force_n_len = length(force_n);
		forces.force += force_n;

        // tangential ELASTIC shear force

		if (tangwall[index*4+3] == 1.0f)
		{
			tangwall[index*4] += tanVel.x * deltaTime;
			tangwall[index*4+1] += tanVel.y * deltaTime;
			tangwall[index*4+2] += tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

		}
		else
		{
			tangwall[index*4] = 0.0f + tanVel.x * deltaTime;
			tangwall[index*4+1] = 0.0f + tanVel.y * deltaTime;
			tangwall[index*4+2] = 0.0f + tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

			tangwall[index*4+3] = 1.0f;

		}


		//force_s += params.bshear*tanVel;						  // Tangential damping (sliding friction)

		float force_s_len = length(force_s);
		if (force_s_len > 0) {
		force_s = force_s / force_s_len;
		}
		else {
		force_s.x = force_s.y = force_s.z = 0.0f;
		}
		force_s = -fminf(force_s_len, params.bcoloumb * force_n_len) * force_s;   // Coloumb sliding friction law

		forces.force += force_s;

		// torque
		forces.torque += radiusA * cross(norm, force_s);

        // attraction
        forces.force += attraction*Pos;
    }

	else
	{
		tangwall[index*4] = 0.0f;
		tangwall[index*4+1] = 0.0f;
		tangwall[index*4+2] = 0.0f;
		tangwall[index*4+3] = 0.0f;
	}

    return forces;
}
__device__
Forces collide2dback(uint index,
					  float3 posA,
                      float3 velA,
					  float massA,
                      float radiusA,
					  float3 omegaA,
                      float attraction,
					  float *tangwall,
					  float deltaTime)
{
    // calculate relative position
    float3 Pos = posA;

    float dist = Pos.x;
    float collideDist = radiusA;

	Forces forces;
	forces.force = make_float3(0.0f);
	forces.torque = make_float3(0.0f);
	float3 force_n = make_float3(0.0f);
	float3 force_s = make_float3(0.0f);
	float3 tangdisp = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm;
		norm.x = 1.0f;
		norm.y = 0;
		norm.z = 0;

        // relative velocity
        float3 relVel = velA;

		// normal relative velocity vector

		float3 relVeln = dot(relVel, norm) * norm;

        // relative tangential velocity
        float3 tanVel = relVel - relVeln - radiusA * cross(omegaA, norm);

        // spring force
        force_n = params.bspring * massA * (collideDist - dist) * norm;

        // dashpot (damping) force
        force_n += -2.0f*params.bdamping * massA * relVeln;

		float force_n_len = length(force_n);
		forces.force += force_n;

		//force_s += params.bshear*tanVel;						  // Tangential damping (sliding friction)

		// tangential ELASTIC shear force

		if (tangwall[index*4+3] == 1.0f)
		{
			tangwall[index*4] += tanVel.x * deltaTime;
			tangwall[index*4+1] += tanVel.y * deltaTime;
			tangwall[index*4+2] += tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

		}
		else
		{
			tangwall[index*4] = 0.0f + tanVel.x * deltaTime;
			tangwall[index*4+1] = 0.0f + tanVel.y * deltaTime;
			tangwall[index*4+2] = 0.0f + tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

			tangwall[index*4+3] = 1.0f;

		}

		float force_s_len = length(force_s);
		if (force_s_len > 0) {
		force_s = force_s / force_s_len;
		}
		else {
		force_s.x = force_s.y = force_s.z = 0.0f;
		}
		force_s = -fminf(force_s_len, params.bcoloumb * force_n_len) * force_s;   // Coloumb sliding friction law

		forces.force += force_s;

		// torque
		forces.torque += radiusA * cross(norm, force_s);

        // attraction
        forces.force += attraction*Pos;
    }
	else
	{
		tangwall[index*4] = 0.0f;
		tangwall[index*4+1] = 0.0f;
		tangwall[index*4+2] = 0.0f;
		tangwall[index*4+3] = 0.0f;
	}

    return forces;
}
__device__
Forces collide2dleftside(uint index,
					  float3 posA,
                      float3 velA,
					  float massA,
                      float radiusA,
					  float3 omegaA,
                      float attraction,
					  float *tangwall,
					  float deltaTime)
{
    // calculate relative position
    float3 Pos = posA;

    float dist = Pos.z;
    float collideDist = radiusA;

	Forces forces;
	forces.force = make_float3(0.0f);
	forces.torque = make_float3(0.0f);
	float3 force_n = make_float3(0.0f);
	float3 force_s = make_float3(0.0f);
	float3 tangdisp = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm;
		norm.x = 0;
		norm.y = 0;
		norm.z = 1.0f;

        // relative velocity
        float3 relVel = velA;

		// normal relative velocity vector
		float3 relVeln = dot(relVel, norm) * norm;

        // relative tangential velocity
        float3 tanVel = relVel - relVeln - radiusA * cross(omegaA, norm);

        // spring force
        force_n = params.bspring * massA * (collideDist - dist) * norm;

        // dashpot (damping) force
        force_n += -2.0f*params.bdamping * massA * relVeln;

		float force_n_len = length(force_n);
		forces.force += force_n;

        // tangential shear force
		//force_s += params.bshear*tanVel;						  // Tangential damping (sliding friction)

		// tangential ELASTIC shear force

		if (tangwall[index*4+3] == 1.0f)
		{
			tangwall[index*4] += tanVel.x * deltaTime;
			tangwall[index*4+1] += tanVel.y * deltaTime;
			tangwall[index*4+2] += tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

		}
		else
		{
			tangwall[index*4] = 0.0f + tanVel.x * deltaTime;
			tangwall[index*4+1] = 0.0f + tanVel.y * deltaTime;
			tangwall[index*4+2] = 0.0f + tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

			tangwall[index*4+3] = 1.0f;

		}

		float force_s_len = length(force_s);
		if (force_s_len > 0) {
		force_s = force_s / force_s_len;
		}
		else {
		force_s.x = force_s.y = force_s.z = 0.0f;
		}
		force_s = -fminf(force_s_len, params.bcoloumb * force_n_len) * force_s;   // Coloumb sliding friction law

		forces.force += force_s;

		// torque
		forces.torque += radiusA * cross(norm, force_s);

        // attraction
        forces.force += attraction*Pos;
    }
	else
	{
		tangwall[index*4] = 0.0f;
		tangwall[index*4+1] = 0.0f;
		tangwall[index*4+2] = 0.0f;
		tangwall[index*4+3] = 0.0f;
	}

    return forces;
}
__device__
Forces collide2drightside(uint index,
					  float3 posA,
                      float3 velA,
					  float massA,
                      float radiusA,
					  float3 omegaA,
                      float attraction,
					  float *tangwall,
					  float deltaTime)
{
    // calculate relative position
    float3 Pos = posA;

    float dist = Pos.z;
    float collideDist = 0.441474f - radiusA;

	Forces forces;
	forces.force = make_float3(0.0f);
	forces.torque = make_float3(0.0f);
	float3 force_n = make_float3(0.0f);
	float3 force_s = make_float3(0.0f);
	float3 tangdisp = make_float3(0.0f);

    if (dist > collideDist)
    {
        float3 norm;
		norm.x = 0;
		norm.y = 0;
		norm.z = -1.0f;

        // relative velocity
        float3 relVel = velA;

		// normal relative velocity vector
		float3 relVeln = dot(relVel, norm) * norm;

        // relative tangential velocity
        float3 tanVel = relVel - relVeln - radiusA * cross(omegaA, norm);

        // spring force
        force_n = params.bspring * massA * (dist - collideDist) * norm;

        // dashpot (damping) force
        force_n += -2.0f*params.bdamping * massA * relVeln;

		float force_n_len = length(force_n);
		forces.force += force_n;

        // tangential shear force
		//force_s += params.bshear*tanVel;						  // Tangential damping (sliding friction)

		// tangential ELASTIC shear force

		if (tangwall[index*4+3] == 1.0f)
		{
			tangwall[index*4] += tanVel.x * deltaTime;
			tangwall[index*4+1] += tanVel.y * deltaTime;
			tangwall[index*4+2] += tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

		}
		else
		{
			tangwall[index*4] = 0.0f + tanVel.x * deltaTime;
			tangwall[index*4+1] = 0.0f + tanVel.y * deltaTime;
			tangwall[index*4+2] = 0.0f + tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

			tangwall[index*4+3] = 1.0f;

		}

		float force_s_len = length(force_s);
		if (force_s_len > 0) {
		force_s = force_s / force_s_len;
		}
		else {
		force_s.x = force_s.y = force_s.z = 0.0f;
		}
		force_s = -fminf(force_s_len, params.bcoloumb * force_n_len) * force_s;   // Coloumb sliding friction law

		forces.force += force_s;

		// torque
		forces.torque += radiusA * cross(norm, force_s);

        // attraction
        forces.force += attraction*Pos;
    }
	else
	{
		tangwall[index*4] = 0.0f;
		tangwall[index*4+1] = 0.0f;
		tangwall[index*4+2] = 0.0f;
		tangwall[index*4+3] = 0.0f;
	}

    return forces;
}
// collide a sphere with bottom wall using DEM method
__device__
Forces collideBottomBoundary(uint index,
					  float3 posA,
                      float3 velA,
					  float massA,
                      float radiusA,
					  float3 omegaA,
                      float attraction,
					  float *tangwall,
					  float deltaTime)
{
    // calculate relative position
    float3 Pos = posA;
    float shearVel= 0.0f;

	float dist = Pos.y;
    float collideDist = radiusA;


	Forces forces;
	forces.force = make_float3(0.0f);
	forces.torque = make_float3(0.0f);
	float3 force_n = make_float3(0.0f);
	float3 force_s = make_float3(0.0f);
	float3 tangdisp = make_float3(0.0f);

    if (dist < collideDist)
    {
    	// Normal vector to inclined plane
        float3 norm;
		norm.x = 0;
		norm.y = 1.0f;
		norm.z = 0;

        // relative velocity
        float3 relVel = velA;
        relVel.z-=shearVel;

		// normal relative velocity
		float3 relVeln = dot(relVel, norm) * norm;

        // relative tangential velocity
        float3 tanVel = relVel - relVeln - radiusA * cross(omegaA, norm);

        // spring force
        force_n = params.bspring * massA * (collideDist - dist) * norm;

        // dashpot (damping) force
        force_n += -2.0f * params.bdamping * massA * relVeln;

		float force_n_len = length(force_n);
		forces.force += force_n;

		//force_s += params.bshear*tanVel;						  // Tangential damping (sliding friction)

		// tangential ELASTIC shear force

		if (tangwall[index*4+3] == 1.0f)
		{
			tangwall[index*4] += tanVel.x * deltaTime;
			tangwall[index*4+1] += tanVel.y * deltaTime;
			tangwall[index*4+2] += tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

		}
		else
		{
			tangwall[index*4] = 0.0f + tanVel.x * deltaTime;
			tangwall[index*4+1] = 0.0f + tanVel.y * deltaTime;
			tangwall[index*4+2] = 0.0f + tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

			tangwall[index*4+3] = 1.0f;

		}

		float force_s_len = length(force_s);
		if (force_s_len > 0) {
		force_s = force_s / force_s_len;
		}
		else {
		force_s.x = force_s.y = force_s.z = 0.0f;
		}
		force_s = -fminf(force_s_len, params.bcoloumb * force_n_len) * force_s;   // Coloumb static friction law

		forces.force += force_s;

		// torque
		forces.torque += radiusA * cross(norm, force_s);

        // attraction
        forces.force += attraction*Pos;
    }
	else
	{
		tangwall[index*4] = 0.0f;
		tangwall[index*4+1] = 0.0f;
		tangwall[index*4+2] = 0.0f;
		tangwall[index*4+3] = 0.0f;
	}

    return forces;
}
// collide a sphere with bottom wall using DEM method
__device__
Forces collideTopBoundary(uint index,
					  float3 posA,
                      float3 velA,
					  float massA,
                      float radiusA,
					  float3 omegaA,
                      float attraction,
					  float *tangwall,
					  float deltaTime,
					  uint  stepCount,
					  float	*tnf,
					  float height,
					  float shearRate)
{
    // calculate relative position
    float3 Pos = posA;
    float shearVel= 0.0f;

    if (stepCount > 100000*8.0f) //was 200000
	{
    	shearVel = shearRate*height; //velocity for moving wall
	}

	float dist = Pos.y;
    float collideDist = height-radiusA;


	Forces forces;
	forces.force = make_float3(0.0f);
	forces.torque = make_float3(0.0f);
	float3 force_n = make_float3(0.0f);
	float3 force_s = make_float3(0.0f);
	float3 tangdisp = make_float3(0.0f);

    if (dist > collideDist)
    {
    	// Normal vector to inclined plane
        float3 norm;
		norm.x = 0;
		norm.y = 1.0f;
		norm.z = 0;

        // relative velocity
        float3 relVel = velA;
        relVel.z-=shearVel;

		// normal relative velocity
		float3 relVeln = dot(relVel, norm) * norm;

        // relative tangential velocity
        float3 tanVel = relVel - relVeln - radiusA * cross(omegaA, norm);

        // spring force
        force_n = params.bspring * massA * (collideDist - dist) * norm;

        // dashpot (damping) force
        force_n += -2.0f * params.bdamping * massA * relVeln;

        atomicAdd(&tnf[0],force_n.y); //adds normal force of current contact to total normal force on top wall

		float force_n_len = length(force_n);
		forces.force += force_n;


		//force_s += params.bshear*tanVel;						  // Tangential damping (sliding friction)

		// tangential ELASTIC shear force

		if (tangwall[index*4+3] == 1.0f)
		{
			tangwall[index*4] += tanVel.x * deltaTime;
			tangwall[index*4+1] += tanVel.y * deltaTime;
			tangwall[index*4+2] += tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

		}
		else
		{
			tangwall[index*4] = 0.0f + tanVel.x * deltaTime;
			tangwall[index*4+1] = 0.0f + tanVel.y * deltaTime;
			tangwall[index*4+2] = 0.0f + tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

			tangwall[index*4+3] = 1.0f;

		}

		float force_s_len = length(force_s);
		if (force_s_len > 0) {
		force_s = force_s / force_s_len;
		}
		else {
		force_s.x = force_s.y = force_s.z = 0.0f;
		}
		force_s = -fminf(force_s_len, params.bcoloumb * force_n_len) * force_s;   // Coloumb static friction law

		forces.force += force_s;

		// torque
		forces.torque += radiusA * cross(norm, force_s);

        // attraction
        forces.force += attraction*Pos;
    }
	else
	{
		tangwall[index*4] = 0.0f;
		tangwall[index*4+1] = 0.0f;
		tangwall[index*4+2] = 0.0f;
		tangwall[index*4+3] = 0.0f;
	}

    return forces;
}
__device__
Forces collideGate(uint index,
					  float3 posA,
                      float3 velA,
					  float massA,
                      float radiusA,
					  float3 omegaA,
                      float attraction,
					  float *tangwall,
					  float deltaTime)
{
    // calculate relative position
    float3 Pos = posA;

    float dist = (0.04f-Pos.z);
    float collideDist = radiusA;

	Forces forces;
	forces.force = make_float3(0.0f);
	forces.torque = make_float3(0.0f);
	float3 force_n = make_float3(0.0f);
	float3 force_s = make_float3(0.0f);
	float3 tangdisp = make_float3(0.0f);

    if (dist < collideDist && Pos.y > params.height && Pos.z < (0.04f+2.0f*radiusA))
    {
        float3 norm;
		norm.x = 0;
		norm.y = 0;
		norm.z = -1.0f;

        // relative velocity
        float3 relVel = velA;

		// normal relative velocity vector
		float3 relVeln = dot(relVel, norm) * norm;

        // relative tangential velocity
        float3 tanVel = relVel - relVeln - radiusA * cross(omegaA, norm);

        // spring force
        force_n = params.bspring * massA * (collideDist - dist) * norm;

        // dashpot (damping) force
        force_n += -params.bdamping * massA * relVeln;

		float force_n_len = length(force_n);
		forces.force += force_n;

        // tangential shear force
		//force_s += params.bshear*tanVel;						  // Tangential damping (sliding friction)

		// tangential ELASTIC shear force

		if (tangwall[index*4+3] == 1.0f)
		{
			tangwall[index*4] += tanVel.x * deltaTime;
			tangwall[index*4+1] += tanVel.y * deltaTime;
			tangwall[index*4+2] += tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

		}
		else
		{
			tangwall[index*4] = 0.0f + tanVel.x * deltaTime;
			tangwall[index*4+1] = 0.0f + tanVel.y * deltaTime;
			tangwall[index*4+2] = 0.0f + tanVel.z * deltaTime;

			tangdisp.x = tangwall[index*4];
			tangdisp.y = tangwall[index*4+1];
			tangdisp.z = tangwall[index*4+2];

			force_s = params.tspring * massA * tangdisp;
			force_s += 2.0f * params.tdamping * massA *tanVel;

			tangwall[index*4+3] = 1.0f;

		}

		float force_s_len = length(force_s);
		if (force_s_len > 0) {
		force_s = force_s / force_s_len;
		}
		else {
		force_s.x = force_s.y = force_s.z = 0.0f;
		}
		force_s = -fminf(force_s_len, params.bcoloumb * force_n_len) * force_s;   // Coloumb sliding friction law

		forces.force += force_s;

		// torque
		forces.torque += radiusA * cross(norm, force_s);

        // attraction
        forces.force += attraction*Pos;
    }
	else
	{
		tangwall[index*4] = 0.0f;
		tangwall[index*4+1] = 0.0f;
		tangwall[index*4+2] = 0.0f;
		tangwall[index*4+3] = 0.0f;
	}

    return forces;
}

__global__
void collideD(float4 *newAcl,               // output: new (unsorted) acceleration
			  float4 *newOmegAcl,           // output: new (unsorted) angular acceleration
              float4 *sortedPos,            // input: sorted positions
			  float4 *unsortedPos,			// input: unsorted positions
              float4 *sortedVel,            // input: sorted velocities
			  float4 *unsortedVel,			// input/output
			  float  *sortedMass,			// input: sorted masses
			  float  *unsortedMass,
			  float4 *sortedVelm,           // input: sorted intermediate velocities
			  float4 *sortedOmegVel,        // input: sorted angular velocities
			  float4 *unsortedOmegVel,
			  float4 *sortedOmegVelm,		// input: sorted intermediate angular velocities
              uint   *gridParticleIndex,    // input: sorted particle indices
			  float  *newNeighborList,		// input: updated list of particle neighbors
			  uint	 *numNeighbors,			// input: array that lists number of neighbors for each particle
			  float  *tangwallfront,		// input: sorted list of particle-wall tangdisps
			  float  *tangwallback,			// input: sorted list of particle-wall tangdisps
			  float  *tangwallleft,			// input: sorted list of particle-wall tangdisps
			  float  *tangwallright,		// input: sorted list of particle-wall tangdisps
			  float  *tangwallbottom,		// input: sorted list of particle-wall tangdisps
			  float  *tangwalltop,          // input: sorted list of particle-wall tangdisps
			  uint   *purgatory,
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles,
			  float   deltaTime,
			  uint    stepCount,
			  float	  *totalNormalForce,
			  float	  *h_totalNormalForce,
			  float   height,
			  float   topWallMass,
			  float   shearRate)
{

    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

	uint originalIndex = gridParticleIndex[index];
	uint numberOfNeighbors = numNeighbors[originalIndex];


    // Skip the force calculation for particles in purgatory
    //if (purgatory[originalIndex] != 0xffffffff) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(FETCH(sortedPos, index));
	float radiusA = FETCH(sortedPos,index).w;
	float mass = FETCH(sortedMass,index);
	float4 vela = FETCH(sortedVel,index);
	float3 vel = make_float3(vela);
	float3 velm = make_float3(FETCH(sortedVelm,index));
	float4 omegvela = FETCH(sortedOmegVel,index);
	float3 omegvel = make_float3(omegvela);
	float3 omegvelm = make_float3(FETCH(sortedOmegVelm, index));

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
   Forces forces;
   forces.force = make_float3(0.0f);
   forces.torque = make_float3(0.0f);
   float3 force = make_float3(0.0f);
   float3 torque = make_float3(0.0f);

   size_t rowIndex = originalIndex*params.pitchNew/sizeof(float);

   for (uint i=0; i < numberOfNeighbors; i++)
   {
	   uint nebId = newNeighborList[rowIndex + 4*i + 0];
	   if(nebId < numParticles)
	   {
		   float3 nebPos = make_float3(FETCH(unsortedPos, nebId));
		   float3 nebVel = make_float3(FETCH(unsortedVel, nebId));
		   float3 nebOmegVel = make_float3(FETCH(unsortedOmegVel, nebId));
		   float nebRadius = FETCH(unsortedPos, nebId).w;
		   float nebMass = FETCH(unsortedMass, nebId);

		   forces = collideSpheres(pos, nebPos, vel, nebVel, mass, nebMass, omegvel, nebOmegVel, radiusA, nebRadius, params.attraction, originalIndex, i, newNeighborList, deltaTime);
		   force += forces.force;
		   torque += forces.torque;
	   }
   }

	__syncthreads();

	forces = collideBottomBoundary(originalIndex, pos, vel, mass, radiusA, omegvel, params.attraction, tangwallbottom, deltaTime);
	force += forces.force;
	torque += forces.torque;

	if (stepCount>0.2*40000.0f) //&& stepCount<1.0*320000.0f)
	{
		forces = collideTopBoundary(originalIndex, pos, vel, mass, radiusA, omegvel, params.attraction, tangwalltop, deltaTime, stepCount, totalNormalForce, height, shearRate);
		force += forces.force;
		torque += forces.torque;
	}
	//force += params.gravity * mass;
	if (stepCount < 0.5*40000.0f)
	{	force += params.gravity * mass;}//////////////////////////////////////////////////////////////////////////////


	if (stepCount>0.5*40000.0f)
	{	force += params.gravity * mass;}//////////////////////////////////////////////////////////////////////////////

	if (stepCount > 1.0*40000.0f)
	{
		float kNudge = 0.025f+0.025f*((topWallMass+0.34315*(height-unsortedPos[originalIndex].y)/height)/(0.34315f*2.0f)); //proportional "nudge coefficient" for enforcing velocity profile
//		float kNudge = 0.02f-0.014f*(height-unsortedPos[originalIndex].y)/height; //proportional "nudge coefficient" for enforcing velocity profile
		force.z += kNudge *((  (unsortedPos[originalIndex].y) * shearRate)-(unsortedVel[originalIndex].z));//top moving, imposed shear
	}

	//Old algorithm
//	newAcl[originalIndex] = make_float4(force / mass, 0.0f);
//	unsortedVel[originalIndex] = make_float4(velm, vela.w) + 0.5f*newAcl[originalIndex] * deltaTime;
//	newOmegAcl[originalIndex] = make_float4((-torque / (0.4f * mass * radiusA * radiusA)), 0.0f);
//	unsortedOmegVel[originalIndex] = make_float4(omegvelm, omegvela.w) + 0.5f * newOmegAcl[originalIndex] * deltaTime;

	//	//This is the new upwinding velocity-updating algorithms
	newAcl[originalIndex] = make_float4(force / mass, 0.0f);
	unsortedVel[originalIndex] = make_float4(velm,vela.w);
	newOmegAcl[originalIndex] = make_float4((-torque / (0.4f * mass * radiusA * radiusA)), 0.0f);
	unsortedOmegVel[originalIndex] = make_float4(omegvelm, omegvela.w);

	if (stepCount == 100000*32.0f)
		{
			unsortedVel[originalIndex].z= (unsortedPos[originalIndex].y)*shearRate;
		}

	// collide with boundary wall using DEM method

	//forces = collide2dfront(originalIndex, pos, vel, mass, radiusA, omegvel, params.attraction, tangwallfront, deltaTime);
	//force += forces.force;
	//torque += forces.torque;

	//forces = collide2dback(originalIndex, pos, vel, mass, radiusA, omegvel, params.attraction, tangwallback, deltaTime);
	//force += forces.force;
	//torque += forces.torque;

	//forces = collide2dleftside(originalIndex, pos, vel, mass, radiusA, omegvel, params.attraction, tangwallleft, deltaTime);
	//force += forces.force;
	//torque += forces.torque;

	//forces = collideGate(originalIndex, pos, vel, mass, radiusA, omegvel, params.attraction, tangwallright, deltaTime);
	//force += forces.force;
	//torque += forces.torque;

	//float dist = (pos.y - ((1.0f - pos.z) * tanf(0.418879f))) * cosf(0.418879f);
	//if ((dist <= radiusA) && radiusA < 0.001125f)
	//{
	//	newAcl[originalIndex] = make_float4(0.0f);
	//	unsortedVel[originalIndex] = make_float4(make_float3(0.0f), vela.w);
	//	newOmegAcl[originalIndex] = make_float4(0.0f);
	//	unsortedOmegVel[originalIndex] = make_float4(make_float3(0.0f), omegvela.w);

	//	return;
	//}
	//else
	//{

	//}

    // write new velocity and acceleration back to original unsorted location

}

__global__
void computeNeighborListD(float *d_newList,
						  float *d_currentList,
						  uint *d_numNeighbors,
						  float4 *sortedPos,
						  uint *d_cellStart,
						  uint *d_cellEnd,
						  uint *gridParticleIndex,
						  uint numParticles)

{
	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

	//uint numNeighborsNeeded = 0;

	if (index >= numParticles) return;

	float4 data = FETCH(sortedPos, index);
	float3 pos = make_float3(data);
	float radiusA = data.w;
	int3 gridPos = calcGridPos(pos);
	uint gridHash = calcGridHash(gridPos);
	uint pid = FETCH(gridParticleIndex, index);

	uint numNeighbors = 0;

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighborCellPos = gridPos + make_int3(x, y, z);
				uint neighborCellHash = calcGridHash(neighborCellPos);
				uint start = d_cellStart[neighborCellHash];
				uint end = d_cellEnd[neighborCellHash];

				for (uint j = start; j < end; j++)
				{
					float4 nebData = sortedPos[j];
					float3 nebPos = make_float3(nebData);
					float radiusB = nebData.w;
					uint nebId = gridParticleIndex[j];
					int rowIndex_new = pid*params.pitchNew/sizeof(float);
					bool isPriorContact = false;

					if(j != index)
					{
						float3 relPos = pos - nebPos;

						float collideDist = radiusA + radiusB;
						// recalculate the normalized position using the Minimum Image Convention
						while (relPos.z<-0.1408) relPos.z += 0.2816f;
						while (relPos.z >= 0.1408) relPos.z -= 0.2816f;
						while (relPos.x<-0.0176) relPos.x += 0.0352f;
						while (relPos.x >= 0.0176) relPos.x -= 0.0352f;

//						while (relPos.z<-0.0704) relPos.z += 0.1408f;
//						while (relPos.z >= 0.0704) relPos.z -= 0.1408f;
//						while (relPos.x<-0.0088) relPos.x += 0.0176f;
//						while (relPos.x >= 0.0088) relPos.x -= 0.0176f;

						float dist = length(relPos);

//						if (dist <= collideDist + 0.5*params.rmax)
//						{
							if (numNeighbors < params.maxNeighbors) // In future, should add a variable maxNeighbors for conditional checking
							{
								if (dist < collideDist)
								{
									int rowIndex_current = pid*params.pitchCurrent/sizeof(float);


									for (uint k = 0; k < d_numNeighbors[pid]; k++)
									{
										float testId = d_currentList[rowIndex_current + 4*k];
										if (nebId == testId)
										{

											d_newList[rowIndex_new + 4*numNeighbors + 0] = testId;
											d_newList[rowIndex_new + 4*numNeighbors + 1] = d_currentList[rowIndex_current + 4*k + 1]; // Need to have some duplicate copy of the contact (neighbor) list to avoid overwriting data before it is needed in the check and compare step!
											d_newList[rowIndex_new + 4*numNeighbors + 2] = d_currentList[rowIndex_current + 4*k + 2];
											d_newList[rowIndex_new + 4*numNeighbors + 3] = d_currentList[rowIndex_current + 4*k + 3];

											isPriorContact = true;
											break;
										}
									}
									if (!isPriorContact)
									{
										d_newList[rowIndex_new + 4*numNeighbors + 0] = nebId;
										d_newList[rowIndex_new + 4*numNeighbors + 1] = 0.0f;
										d_newList[rowIndex_new + 4*numNeighbors + 2] = 0.0f;
										d_newList[rowIndex_new + 4*numNeighbors + 3] = 0.0f;
									}
									numNeighbors++;
								}
							}
//						}
					}
				}
			}
		}
	}

	d_numNeighbors[pid] = numNeighbors;

}

#endif
