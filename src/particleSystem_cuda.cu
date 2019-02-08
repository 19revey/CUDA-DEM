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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "thrust/device_vector.h"
#include "thrust/extrema.h"

#include "particles_kernel_impl.cuh"

struct is_purged : thrust::unary_function<uint, bool>
{
    __host__ __device__
    bool operator()(const uint &x)
    {
        return x != 0xffffffff;
    }
};

struct purgeReset
{
    __host__ __device__
    void operator()(uint &x)
    {
        x = 0xffffffff;
    }
};

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void cudaGLInit(int argc, char **argv)
    {
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        findCudaGLDevice(argc, (const char **)argv);
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

	void allocate2dArray(void **devPtr, size_t* pitch, size_t width, size_t height)
	{
		checkCudaErrors(cudaMallocPitch(devPtr, pitch, width, height));
	}

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

	void copy2dArrayToDevice(void *device, size_t dpitch, const void *host, size_t spitch, size_t width, size_t height)
	{
		checkCudaErrors(cudaMemcpy2D(device, dpitch, host, spitch, width, height, cudaMemcpyHostToDevice));
	}

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsNone));
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
    {
        void *ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
                                                             *cuda_vbo_resource));
        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }

    void copyArrayFromDevice(void *host, const void *device,
                             struct cudaGraphicsResource **cuda_vbo_resource, int size)
    {
        if (*cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

        if (*cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

	void copy2dArrayFromDevice(void *host, size_t spitch, const void *device, size_t dpitch, size_t width, size_t height)
	{
		checkCudaErrors(cudaMemcpy2D(host, spitch, device, dpitch, width, height, cudaMemcpyDeviceToHost));
	}

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void integrateSystem(float *pos,
                         float *vel,
						 float *acl,
						 float *velm,
						 float *omegvel,
						 float *omegacl,
						 float *omegvelm,
                         float deltaTime,
                         uint numParticles)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);
		thrust::device_ptr<float4> d_acl4((float4 *)acl);
		thrust::device_ptr<float4> d_velm4((float4 *)velm);
		thrust::device_ptr<float4> d_omegvel4((float4 *)omegvel);
		thrust::device_ptr<float4> d_omegacl4((float4 *)omegacl);
		thrust::device_ptr<float4> d_omegvelm4((float4 *)omegvelm);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_acl4, d_velm4, d_omegvel4, d_omegacl4, d_omegvelm4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles, d_acl4+numParticles, d_velm4+numParticles, d_omegvel4+numParticles, d_omegacl4+numParticles, d_omegvelm4+numParticles)),
            integrate_functor(deltaTime));
    }

	void periodicBoundaries(float *pos,
							float maxZ,
							float maxX,
							uint numParticles)
	{
		thrust::device_ptr<float4> d_pos4((float4 *)pos);
		thrust::for_each(d_pos4,d_pos4+numParticles,periodic_functor(maxZ,maxX));
	}

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
						 float smallParticleRadius)
	{

		//printf("Purgatory called.\n");
		// A device vector that collects all the particle id's in purgatory
		thrust::device_vector<uint> miscreants(numParticles);

		// A device vector to hold only the number of particles' id's we want to reintroduce
		thrust::device_vector<uint> releasedParticles(number);

		// Convert raw pointer into Thrust device_ptr
		thrust::device_ptr<uint> dPurg((uint *)purgatory);

		thrust::copy_if(dPurg,dPurg+numParticles,miscreants.begin(),is_purged());
		thrust::copy_n(miscreants.begin(), number, releasedParticles.begin());

		//printf("Copy success.\n");

		// Release the particles from purgatory by performing a scattered write
		thrust::for_each(thrust::make_permutation_iterator(dPurg,releasedParticles.begin()),
						 thrust::make_permutation_iterator(dPurg,releasedParticles.end()),purgeReset());

		//printf("Particle release success.\n");

		// Now re-introduce the released particles back into the simulation at the feed zone, potentially changing their attributes
		// as needed. //

		thrust::device_ptr<float4> Pos((float4 *)pos);
		thrust::device_ptr<float4> Velm((float4 *)velm);
		thrust::device_ptr<float4> OmegVel((float4 *)omegvel);
		thrust::device_ptr<float4> OmegVelm((float4 *)omegvelm);
		thrust::device_ptr<float4> OmegAcl((float4 *)omegacl);
		thrust::device_ptr<float4> Acl((float4 *)acl);
		thrust::device_ptr<float4> Vel((float4 *)vel);
		thrust::device_ptr<float> Mass((float *)mass);
		thrust::device_ptr<float4> Col((float4 *)color);

		// Create a temporary arrangement of particles in the feed zone in host memory to be transferred to the device
		// through a scattered write

		uint i = 0;

		//printf("Beginning feed configuration.\n");

//		initializeDistribution(number);
//
//		float *colorPtr = new float[3];
//
//		for(int x=0;x<feedSpacing.x;x++)
//		{
//			for(int g=feedSpacing.y;g>0;g--)
//			{
//				for(int z=0;z<feedSpacing.z;z++)
//				{
//					if (i<number)
//					{
//						hpos[(i)*4] = (2*largeParticleRadius * 1.0f * x) + largeParticleRadius + (frand()*2.0f-1.0f)*0.001f;
//						hpos[(i)*4+1] = (2*largeParticleRadius * g) + largeParticleRadius + (0.33f-maxDropHeight) + (frand()*2.0f-1.0f)*0.001f;
//						hpos[(i)*4+2] = (2*largeParticleRadius * 1.0f * z) + largeParticleRadius + (frand()*2.0f-1.0f)*0.001f;
//
//						hpos[(i)*4+3] = m_hRadiiTemp[i];
//
//						hvelm[(i)*4] = 0.0f;
//						hvelm[(i)*4+1] = 0.0f;
//						hvelm[(i)*4+2] = 0.0f;
//						hvelm[(i)*4+3] = 0.0f;
//
//						homegvel[(i)*4] = 0.0f;
//						homegvel[(i)*4+1] = 0.0f;
//						homegvel[(i)*4+2] = 0.0f;
//						homegvel[(i)*4+3] = (float)releasedParticles[i];
//
//						homegacl[(i)*4] = 0.0f;
//						homegacl[(i)*4+1] = 0.0f;
//						homegacl[(i)*4+2] = 0.0f;
//						homegacl[(i)*4+3] = 0.0f;
//
//						hacl[(i)*4] = 0.0f;
//						hacl[(i)*4+1] = 0.0f;
//						hacl[(i)*4+2] = 0.0f;
//						hacl[(i)*4+3] = 0.0f;
//
//						homegvelm[(i)*4] = 0.0f;
//						homegvelm[(i)*4+1] = 0.0f;
//						homegvelm[(i)*4+2] = 0.0f;
//						homegvelm[(i)*4+3] = 0.0f;
//
//						hvel[(i)*4] = 0.0f;
//						hvel[(i)*4+2] = 0.0f;
//						hvel[(i)*4+3] = 1.0f;
//
//						colorRamp(hpos[(i)*4+3],colorPtr);
//						hcolor[(i)*4] = colorPtr[0];
//						hcolor[(i)*4+1] = colorPtr[1];
//						hcolor[(i)*4+2] = colorPtr[2];
//						hcolor[(i)*4+3] = 1.0f;
//
//						hmass[(i)] = 4.0f / 3.0f * CUDART_PI_F * powf(hpos[(i)*4+3],3) * 2500.0f;
//
//						hvel[(i)*4+1] = -sqrtf(powf(initVelocity.y,2.0f)-2.0f*m_params.gravity.y*(maxDropHeight - 2.0f*largeParticleRadius * g));
//
//						i++;
//					}
//				}
//			}
//		}

//		for(int x=0;x<feedSpacing.x;x++)
//		{
//			for(int g=0;g<feedSpacing.y;g++)
//			{
//				for(int z=0;z<feedSpacing.z;z++)
//				{
//					if (i<number)
//					{
//						hpos[i*4] = (2*largeParticleRadius * x) + largeParticleRadius + (frand()*2.0f-1.0f)*0.001f;
//						hpos[i*4+1] = (2*largeParticleRadius * g) + largeParticleRadius + 0.50290f + (frand()*2.0f-1.0f)*0.001f;
//						hpos[i*4+2] = (2*largeParticleRadius * z) + largeParticleRadius +0.001f + (frand()*2.0f-1.0f)*0.001f;
//						hpos[i*4+3] = 1.0f;
//
//						hvelm[i*4] = 0.0f;
//						hvelm[i*4+1] = 0.0f;
//						hvelm[i*4+2] = 0.0f;
//						hvelm[i*4+3] = 0.0f;
//
//						homegvel[i*4] = 0.0f;
//						homegvel[i*4+1] = 0.0f;
//						homegvel[i*4+2] = 0.0f;
//
//						// Be careful not to mess up the particle id's here
//						homegvel[i*4+3] = (float)releasedParticles[i];
//
//						homegacl[i*4] = 0.0f;
//						homegacl[i*4+1] = 0.0f;
//						homegacl[i*4+2] = 0.0f;
//						homegacl[i*4+3] = 0.0f;
//
//						hacl[i*4] = 0.0f;
//						hacl[i*4+1] = 0.0f;
//						hacl[i*4+2] = 0.0f;
//						hacl[i*4+3] = 0.0f;
//
//						homegvelm[i*4] = 0.0f;
//						homegvelm[i*4+1] = 0.0f;
//						homegvelm[i*4+2] = 0.0f;
//						homegvelm[i*4+3] = 0.0f;
//
//						hvel[i*4] = 0.0f;
//						hvel[i*4+1] = 0.0f;
//						hvel[i*4+2] = 0.0f;
//						hvel[i*4+3] = 1.0f;
//
//						hcolor[i*4] = 1.0f;
//						hcolor[i*4+1] = 0.2f;
//						hcolor[i*4+2] = 0.2f;
//						hcolor[i*4+3] = 1.0f;
//
//						hmass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(hpos[i*4+3],3) * 2500.0f;
//
//						i++;
//
//					}
//				}
//			}
//		}

		printf("Recycle i = %i\n",i);

		i = 0;

		float sizeratio = powf(largeParticleRadius/smallParticleRadius,3.0f);
		float nratio = 0.575f / (1.0f - 0.575f) * (1.0f / sizeratio);
		float ratio = nratio / (1 + nratio);

		//printf("Ratio: %f\n",ratio);

		do
		{
			float h = frand();
				if (hvel[int(number * h)*4+3] == 1.0f) {
					hvel[int(number * h)*4+3] = 2.0f;
					i++;
				}
		}
		while (i < int(number * ratio));

		printf("Particle type change success.\n");

		i = 0;
		do
		{
			float h = frand();
			if (hvel[i*4+3] == 1.0f){
				hpos[i*4+3] = (2.0f*h - 1.0f) * 0.1f*smallParticleRadius + smallParticleRadius;
				hcolor[i*4] = 1.0;
				hcolor[i*4+1] = 0.2;   // Particle colors
				hcolor[i*4+2] = 0.2;
				hcolor[i*4+3] = 1.0;
			}
			else {
				hpos[i*4+3] = (2.0f*h - 1.0f) * 0.1f*largeParticleRadius + largeParticleRadius;
				hcolor[i*4] = 0.2;
				hcolor[i*4+1] = 0.2;    // Particle colors
				hcolor[i*4+2] = 1.0;
				hcolor[i*4+3] = 1.0;

			}
			hmass[i] = 4.0f / 3.0f * CUDART_PI_F * powf(hpos[i*4+3],3) * 2500.0f;
			i++;
		}
		while (i < (int)number);
		//printf("Particle color type change success.\n");

		// Now perform the scattered write to the existing device arrays

		// Wrap host pointers into Thrust device vectors for easier manipulation

		// First recast host pointers into their natural size

		float4 *t_hPos;
		float4 *t_hVel;
		float4 *t_hVelm;
		float4 *t_hAcl;
		float4 *t_hOmegVel;
		float4 *t_hOmegVelm;
		float4 *t_hOmegAcl;
		float4 *t_hCol;
		float  *t_hMass;

		t_hPos = (float4 *) hpos;
		t_hVel = (float4 *) hvel;
		t_hVelm = (float4 *) hvelm;
		t_hAcl = (float4 *) hacl;
		t_hOmegVel = (float4 *) homegvel;
		t_hOmegVelm = (float4 *) homegvelm;
		t_hOmegAcl = (float4 *) homegacl;
		t_hCol = (float4 *) hcolor;
		t_hMass = (float *) hmass;

		thrust::device_vector<float4> d_t_hPos(t_hPos,t_hPos+number);
		thrust::device_vector<float4> d_t_hVel(t_hVel,t_hVel+number);
		thrust::device_vector<float4> d_t_hVelm(t_hVelm,t_hVelm+number);
		thrust::device_vector<float4> d_t_hAcl(t_hAcl,t_hAcl+number);
		thrust::device_vector<float4> d_t_hOmegVel(t_hOmegVel,t_hOmegVel+number);
		thrust::device_vector<float4> d_t_hOmegVelm(t_hOmegVelm,t_hOmegVelm+number);
		thrust::device_vector<float4> d_t_hOmegAcl(t_hOmegAcl,t_hOmegAcl+number);
		thrust::device_vector<float4> d_t_hCol(t_hCol,t_hCol+number);
		thrust::device_vector<float>  d_t_hMass(t_hMass,t_hMass+number);

		//printf("Performing scattered write...\n");

		thrust::scatter(thrust::make_zip_iterator(thrust::make_tuple(d_t_hPos.data(),d_t_hVel.data(),d_t_hVelm.data(),d_t_hAcl.data(),d_t_hOmegVel.data(),d_t_hOmegVelm.data(),d_t_hOmegAcl.data(),d_t_hCol.data(),d_t_hMass.data())),
						thrust::make_zip_iterator(thrust::make_tuple(d_t_hPos.data()+number,d_t_hVel.data()+number,d_t_hVelm.data()+number,d_t_hAcl.data()+number,d_t_hOmegVel.data()+number,d_t_hOmegVelm.data()+number,d_t_hOmegAcl.data()+number,d_t_hCol.data()+number,d_t_hMass.data()+number)),
						releasedParticles.begin(),thrust::make_zip_iterator(thrust::make_tuple(Pos,Vel,Velm,Acl,OmegVel,OmegVelm,OmegAcl,Col,Mass)));

		//printf("Scattered write success.\n");

	}

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  uint    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (float4 *) pos,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: calcHash");
    }

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
									 uint  *d_numNeighbors,
                                     float *oldPos,
                                     float *oldVel,
									 float *oldMass,
									 float *oldVelm,
									 float *oldOmegVel,
									 float *oldOmegVelm,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelmTex, oldVelm, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldOmegVelTex, oldOmegVel, numParticles*sizeof(float4)));
#endif

        uint smemSize = sizeof(uint)*(numThreads+1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(cellStart,
																		   cellEnd,
																		   (float4 *) sortedPos,
																		   (float4 *) sortedVel,
																		   (float *) sortedMass,
																		   (float4 *) sortedVelm,
																		   (float4 *) sortedOmegVel,
																		   (float4 *) sortedOmegVelm,
																		   sortedNumNeighbors,
																		   gridParticleHash,
																		   gridParticleIndex,
																		   d_numNeighbors,
																		   (float4 *) oldPos,
																		   (float4 *) oldVel,
																		   (float *)  oldMass,
																		   (float4 *) oldVelm,
																		   (float4 *) oldOmegVel,
																		   (float4 *) oldOmegVelm,
																		   numParticles);

        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
		checkCudaErrors(cudaUnbindTexture(oldVelmTex));
		checkCudaErrors(cudaUnbindtexture(oldOmegVelTex));
#endif
    }

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
							 uint maxNeighbors)

	{
		uint numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		checkCudaErrors(cudaMemcpy2D(d_currentList, pitch_current, d_newList, pitch_new, sizeof(float)*4*maxNeighbors, numParticles, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemset2D(d_newList, pitch_new, 0, sizeof(float)*4*maxNeighbors, numParticles)); // Re-initialize the new 2D neighbor list

		dim3 blocksPerGrid(4,64,96);

		computeNeighborListD<<< numBlocks, numThreads >>>(d_newList,
													 d_currentList,
													 d_numNeighbors,
													 (float4 *)d_Pos,
													 d_cellStart,
													 d_cellEnd,
													 gridParticleIndex,
													 numParticles);

		getLastCudaError("Neighbor list generation failed.");
	}

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
				 uint *purgatory,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells,
				 float  deltaTime,
				 uint   stepCount,
				 float  *totalNormalForce,
				 float  *h_totalNormalForce,
				 float  height,
				 float  topWallMass,
				 float  shearRate)
    {
#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelmTex, sortedVelm, numParticles*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldOmegVelTex, sortedOmegVel, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
        checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

		//uint smemSize = 16*4*sizeof(float)*256;

        // execute the kernel
        collideD<<< numBlocks, numThreads >>>((float4 *)newAcl,                 // Notice here the casting to float4 pointers from float pointers in the parameter list.
											  (float4 *)newOmegAcl,
                                              (float4 *)sortedPos,
											  (float4 *)unsortedPos,
                                              (float4 *)sortedVel,
											  (float4 *)unsortedVel,
											  (float *)sortedMass,
											  (float *)unsortedMass,
											  (float4 *)sortedVelm,
											  (float4 *)sortedOmegVel,
											  (float4 *)unsortedOmegVel,
											  (float4 *)sortedOmegVelm,
                                              gridParticleIndex,
											  newNeighborList,
											  numNeighbors,
											  tangwallfront,
											  tangwallback,
											  tangwallleft,
											  tangwallright,
											  tangwallbottom,
											  tangwalltop,
											  purgatory,
                                              cellStart,
                                              cellEnd,
                                              numParticles,
											  deltaTime,
											  stepCount,
											  totalNormalForce,
											  h_totalNormalForce,
											  height,
											  topWallMass,
											  shearRate);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed: collide");
//        printf("Total Normal Force d %f\n",totalNormalForce);
//        copyArrayFromDevice(h_totalNormalForce,totalNormalForce,0,sizeof(float));
        cudaMemcpy(h_totalNormalForce, totalNormalForce, sizeof(float), cudaMemcpyDeviceToHost);

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
		checkCudaErrors(cudaUnbindTexture(oldVelmTex));
		checkCudaErrors(cudaUnbindTexture(oldOmegVelTex));
        checkCudaErrors(cudaUnbindTexture(cellStartTex));
        checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
    }


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }

}   // extern "C"
