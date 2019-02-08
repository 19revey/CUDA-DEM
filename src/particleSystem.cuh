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

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
	void allocate2dArray(void **devPtr, size_t * pitch, size_t width, size_t height);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
	void copy2dArrayFromDevice(void *host, size_t spitch, const void *device, size_t dpitch, size_t width, size_t height);
	void copy2dArrayToDevice(void *device, size_t dpitch, const void *host, size_t spitch, size_t width, size_t height);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


    void setParameters(SimParams *hostParams);

    void integrateSystem(float *pos,
                         float *vel,
						 float *acl,
						 float *velm,
						 float *omegvel,
						 float *omegacl,
						 float *omegvelm,
                         float deltaTime,
                         uint numParticles);

	void periodicBoundaries(float *pos,
							float maxZ,
							float maxX,
							uint numParticles);

	void particleRaptureAndReintroduction(float *pos,
										  float *vel,
										  float *acl,
										  float *velm,
										  float *omegvel,
										  float *omegacl,
										  float *omegvelm,
										  float *color,
										  float *mass,
										  uint  *purgatory,
										  uint3 feedSpacing,
										  uint number,
										  uint numParticles,
										  float *hpos,
										  float *hvel,
										  float *hacl,
										  float *hvelm,
										  float *homegvel,
										  float *homegacl,
										  float *homegvelm,
										  float *hcolor,
										  float *hmass,
										  float largeParticleRadius,
										  float smallParticleRadius);


    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  uint    numParticles);


	void computeNeighborList(float *d_newList,
							 size_t pitch_new,
							 float *d_currentList,
							 size_t pitch_current,
							 uint *d_numNeighbors,
							 float *d_Pos,
							 uint *d_cellStart,
							 uint *d_cellEnd,
							 uint *gridParticleIndex,
							 uint numParticles,
							 uint numCells,
							 uint maxNeighbors);


    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
									 float *sortedMass,
									 float *sortedVelm,
									 float *sortedOmegVel,
									 float *sortedOmegVelm,
									 uint  *sortedNumNeighbors,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
									 uint  *numNeighbors,
                                     float *oldPos,
                                     float *oldVel,
									 float *oldMass,
									 float *oldVelm,
									 float *oldOmegVel,
									 float *oldOmegVelm,
                                     uint   numParticles,
                                     uint   numCells);

    void collide(float *newAcl,
				 float *newOmegAcl,
				 float *sortedPos,
				 float *unsortedPos,
                 float *sortedVel,
                 float *unsortedVel,
				 float *sortedMass,
				 float *unsortedMass,
				 float *sortedVelm,
				 float *sortedOmegVel,
				 float *unsortedOmegVel,
				 float *sortedOmegVelm,
                 uint  *gridParticleIndex,
				 float *newNeighborList,
				 uint  *numNeighbors,
				 float *tangwallfront,
				 float *tangwallback,
				 float *tangwallleft,
				 float *tangwallright,
				 float *tangwallbottom,
				 float *tangwalltop,
				 uint  *purgatory,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells,
				 float  deltaTime,
				 uint   stepCount,
				 float  *totalNormalForce,
				 float  *h_totalNormalForce,
				 float	height,
				 float  topWallMass,
				 float  shearRate);

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

}
