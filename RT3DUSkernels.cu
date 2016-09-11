// System includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
//#include <thrust/fill.h>
//#include <thrust/sequence.h>
//https://github.com/thrust/thrust/blob/master/examples/bounding_box.cu

extern "C"
cudaError_t discretizePoints(const float4 *dev_points, float4 *dev_discPoints, int4 * dev_pointIdx, const int nPoints, const float3 mapRes, const float3 gridMin, const float3 gridMax, const cudaExtent gridDims);

extern "C"
cudaError_t addObsToVolume(const int4 * dev_pointIdx, int * dev_volume, const int nPoints, const cudaExtent gridDims);

extern "C"
cudaError_t computeConvexHull(const int4 * dev_pointIdx, int * dev_volume, const int nPoints, const cudaExtent gridDims);

/**
* Discretizes a given set of points.
* points: Given points. .x .y .z are the coordinates, .w holds the data (0-255).
* discPoints: Discretized points. .x .y .z are the coordinates, .w holds the data.
* pointIdx: Holds the indices for each each point in the 3D grid
* nPoints: Number of given data points.
* mapRes: Resolution of the discretization grid in x,y,z (1 unit in data corresponds to 1/mapRes units in the Voronoi diagram)
* gridMin: Minimum point coords
* gridMax: Maximum point coords // probably not needed
*/
__global__ void discretizePoints_kernel(const float4 *points, float4 *discPoints, int4 * pointIdx, const int nPoints, const float3 mapRes, const float3 gridMin, const float3 gridMax, const cudaExtent gridDims)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x; //thread index

	if (idx > nPoints) { return; } // thread index greater than number of points in the points vector
	 
	float4 point = points[idx]; // get the point at location idx
	float4 newLoc; // discretized location
	int4 newIdx; // index in the volume
		
	//newIdx.x = (int)roundf((point.x - gridMin.x) / mapRes.x); // account for offset, if there is any
	//newIdx.y = (int)roundf((point.y - gridMin.y) / mapRes.y);
	//newIdx.z = (int)roundf((point.z - gridMin.z) / mapRes.z);
	//newIdx.w = (int)roundf(point.w);

	newIdx.x = (int)roundf((point.x - gridMin.x) / (gridMax.x - gridMin.x) * gridDims.width); // account for offset, if there is any
	newIdx.y = (int)roundf((point.y - gridMin.y) / (gridMax.y - gridMin.y) * gridDims.height);
	newIdx.z = (int)roundf((point.z - gridMin.z) / (gridMax.z - gridMin.z) * gridDims.depth);
	newIdx.w = (int)roundf(point.w);

	newLoc.x = newIdx.x * mapRes.x + gridMin.x; // scale by map resolution and round to nearest integer
	newLoc.y = newIdx.y * mapRes.y + gridMin.y;
	newLoc.z = newIdx.z * mapRes.z + gridMin.z;
	newLoc.w = roundf(point.w); // data value rounded to nearest integer

	pointIdx[idx] = newIdx; // add grid indices to pointIdx vector
	discPoints[idx] = newLoc; // add descretized point to discrete points vector

	__syncthreads(); // we could call cudaDeviceSynchronize instead
}

/**
* Assigns discretized points to their indices. Resolves multiple writes using max.
* pointIdx: Holds the indices for each each point in the 3D grid, and the color (.w)
* volume: Output volume
* nPoints: Number of given data points.
* gridDims: Height x Width x Depth of the map.
*/
__global__ void addObsToVolume_kernel(const int4 * pointIdx, int * volume, const int nPoints, const cudaExtent gridDims)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x; //thread index

	if (idx >= nPoints) // thread index greater than number of points in the pointIdx vector
	{
		return;
	}

	int4 point = pointIdx[idx];

	int linearIdx = gridDims.width*gridDims.height*point.z + gridDims.width*point.y + point.x;
	//int linearIdx = gridDims.width*gridDims.height*point.z + gridDims.height*point.x + point.y;

	int nVoxels = gridDims.width*gridDims.height*gridDims.depth;
	if (linearIdx >= nVoxels)
		printf("--- Error in addObsToVolume_kernel: linearIdx >= nVoxels\n\n");

	atomicMax(&volume[linearIdx], point.w);

	__syncthreads();
}

// Helper function for using CUDA to add vectors in parallel.
extern "C"
cudaError_t discretizePoints(const float4 *dev_points, float4 *dev_discPoints, int4 *dev_pointIdx, int nPoints, const float3 mapRes, const float3 gridMin, const float3 gridMax, const cudaExtent gridDims)
{
	cudaError_t cudaStatus;

	// Calculate grid size
	int blockSize;      // The launch configurator returned block size 
	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;       // The actual grid size needed, based on input size 
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, discretizePoints_kernel, 0, nPoints);

	gridSize = (nPoints + blockSize - 1) / blockSize;

	// Launch a kernel on the GPU with one thread for each element.
	printf("Launching discretizePoints_kernel with <%d, %d> grids/blocks.\n", gridSize, blockSize);
	discretizePoints_kernel <<< gridSize, blockSize >>>(dev_points, dev_discPoints, dev_pointIdx, nPoints, mapRes, gridMin, gridMax, gridDims);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

extern "C"
cudaError_t addObsToVolume(const int4 * dev_pointIdx, int * dev_volume, const int nPoints, const cudaExtent gridDims)
{
	cudaError_t cudaStatus;

	// Calculate grid size
	int blockSize;      // The launch configurator returned block size 
	int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;       // The actual grid size needed, based on input size 
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, addObsToVolume_kernel, 0, nPoints);

	gridSize = (nPoints + blockSize - 1) / blockSize;

	// Launch a kernel on the GPU with one thread for each element.
	printf("Launching addObsToVolume_kernel with <%d, %d> grids/blocks.\n", gridSize, blockSize);
	addObsToVolume_kernel <<< gridSize, blockSize >>>(dev_pointIdx, dev_volume, nPoints, gridDims);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}