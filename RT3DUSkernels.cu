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
cudaError_t interpolateVolume(int * dev_volume, const cudaExtent gridDims);

extern "C"
cudaError_t checkNegOnes(int * dev_volume, const cudaExtent gridDims, int *flag);

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

//	int nVoxels = gridDims.width*gridDims.height*gridDims.depth;
//	if (linearIdx >= nVoxels)
//		printf("--- Error in addObsToVolume_kernel: linearIdx >= nVoxels\n\n");

	atomicMax(&volume[linearIdx], point.w);

	__syncthreads();
}

/**
* Interpolate.
* volume: Output volume
* gridDims: Width x Height x  Depth of the map.
*/
__global__ void interpolate_kernel(int * dev_volume, const cudaExtent gridDims)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; //thread index

    int nPoints = gridDims.width*gridDims.height*gridDims.depth;

    if (idx > nPoints) { return; } // thread index greater than number of points in the volume

    int pixVal = dev_volume[idx]; // get the point at location idx

    if( pixVal != -1 )
        return; // point is outside the interpolation boundary or already has a value

    // compute 3D idx
    int x,y,z, wh;
    wh = gridDims.width*gridDims.height;
    z = idx / wh;
    y = (idx - z*wh) / gridDims.width;
    x = (idx - z*wh - y*gridDims.width);

    // take a look at neighbors to see if there are any values to be used
    int right, left, up, down, front, back, rightLin, leftLin, upLin, downLin, frontLin, backLin;
    right = x+1        ; rightLin =     z*wh +    y*gridDims.width + right;
    left  = x-1        ; leftLin  =     z*wh +    y*gridDims.width +  left;
    up    =     y+1    ; upLin    =     z*wh +   up*gridDims.width +     x;
    down  =     y-1    ; downLin  =     z*wh + down*gridDims.width +     x;
    front =         z+1; frontLin = front*wh +    y*gridDims.width +     x;
    back  =         z-1; backLin  =  back*wh +    y*gridDims.width +     x;

    int rPix = -3, lPix = -3, uPix = -3, dPix = -3, fPix = -3, bPix = -3;
    if( (right < gridDims.width ) && (rightLin < nPoints)) { rPix = dev_volume[rightLin]; }
    if( ( left > -1)                                     ) { lPix = dev_volume[ leftLin]; }
    if( (   up < gridDims.height) && (   upLin < nPoints)) { uPix = dev_volume[   upLin]; }
    if( ( down > -1)                                     ) { dPix = dev_volume[ downLin]; }
    if( (front < gridDims.depth ) && (frontLin < nPoints)) { fPix = dev_volume[frontLin]; }
    if( ( back > -1)                                     ) { bPix = dev_volume[ backLin]; }

    int sum = rPix + lPix + uPix + dPix + fPix + bPix;

    if(sum == -18) // all out of bounds - should not happen unless volume is size 1x1x1
    { dev_volume[idx] = 0; return; }

    if( (rPix == -1) && (lPix == -1) && (uPix == -1) && (dPix == -1) && (fPix == -1) && (bPix == -1) )
    { return; } // do not change -1, we'll come back to it

    if( (rPix == -2) && (lPix == -2) && (uPix == -2) && (dPix == -2) && (fPix == -2) && (bPix == -2) )
    { dev_volume[idx] = 0; return; } // all outside conv hull - should not happen unless we have a single point

    // handle partial cases
    int div = 0;
    if( rPix < 0) { rPix = 0; } else { div++; }
    if( lPix < 0) { lPix = 0; } else { div++; }
    if( uPix < 0) { uPix = 0; } else { div++; }
    if( dPix < 0) { dPix = 0; } else { div++; }
    if( fPix < 0) { fPix = 0; } else { div++; }
    if( bPix < 0) { bPix = 0; } else { div++; }

    if(div < 1) { return; } // messed up, this shouldn't happen
    float newPixVal = (rPix + lPix + uPix + dPix + fPix + bPix)/(float)div;

    dev_volume[idx] = (int)newPixVal;

    __syncthreads(); // we could call cudaDeviceSynchronize instead
}

/**
* Interpolate.
* volume: Output volume
* gridDims: Width x Height x Depth of the map.
* flag : size 1 - 0 if no -1 found, >0 if found
*/
__global__ void checkNegOnes_kernel(int * dev_volume, const cudaExtent gridDims, int *flag)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x; //thread index

    int pixVal = dev_volume[idx];

    if(pixVal == -1)
        atomicAdd(flag, 1);

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

extern "C"
cudaError_t interpolateVolume(int * dev_volume, const cudaExtent gridDims)
{
    cudaError_t cudaStatus;

    int nPoints = gridDims.width*gridDims.height*gridDims.depth;

    // Calculate grid size
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, interpolate_kernel, 0, nPoints);

    gridSize = (nPoints + blockSize - 1) / blockSize;

    int h_flag = 1;
    int *dev_flag;
    // Allocate flag
    printf("CUDA malloc dev_flag.\n");
    cudaStatus = cudaMalloc((void**)&dev_flag, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    int nLoops = 0;
    while( (h_flag > 0) || (nLoops < 1000) )
    {
        // set flag to 0
        printf("CUDA memset dev_flag to 0.\n");
        cudaStatus = cudaMemset(dev_flag, 0, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemset failed!");
            return cudaStatus;
        }

        // Launch a kernel on the GPU with one thread for each element.
        printf("Launching interpolate_kernel with <%d, %d> grids/blocks.\n", gridSize, blockSize);
        interpolate_kernel <<< gridSize, blockSize >>>(dev_volume, gridDims);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "interpolateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = checkNegOnes(dev_volume, gridDims, dev_flag);

        printf("CUDA memcpy dev_flag to h_flag.\n");
        cudaStatus = cudaMemcpy(&h_flag, dev_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return cudaStatus;
        }
        nLoops++;
    }

    return cudaStatus;
}

extern "C"
cudaError_t checkNegOnes(int * dev_volume, const cudaExtent gridDims, int *flag)
{
    cudaError_t cudaStatus;

    int nPoints = gridDims.width*gridDims.height*gridDims.depth;

    // Calculate grid size
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, checkNegOnes_kernel, 0, nPoints);

    gridSize = (nPoints + blockSize - 1) / blockSize;

    // Launch a kernel on the GPU with one thread for each element.
    printf("Launching checkNegOnes_kernel with <%d, %d> grids/blocks.\n", gridSize, blockSize);
    checkNegOnes_kernel <<< gridSize, blockSize >>>(dev_volume, gridDims, flag);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "checkNegOnesKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    return cudaStatus;
}
