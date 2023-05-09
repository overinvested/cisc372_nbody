#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "parallel_compute.h"
#include <unistd.h>


extern vector3 *d_values, **d_accels, *d_hPos, *d_hVel, *d_accel_sum;
extern double *d_mass;


//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	int fillBlocks = (NUMENTITIES+7)/8;
	dim3 fillThreads(8,3);
	fillAccelSum<<<fillBlocks, fillThreads>>>(d_accel_sum);
	int blocksPerDim=(NUMENTITIES+7)/8;
	dim3 calculateThreads(8, 8, 3);
	dim3 calculateBlocks(blocksPerDim,blocksPerDim);
	calculateAccels<<<calculateBlocks, calculateThreads>>>(d_accels, d_hPos, d_mass);
	int sumThreads = 1024;
	dim3 sumBlocks(NUMENTITIES,3);
	int sharedMemorySize = 2*sumThreads*sizeof(double);
	sumColumns<<<sumBlocks, sumThreads, sharedMemorySize>>>(d_accels, d_accel_sum);
	int updateBlocks = (NUMENTITIES+7)/8;
	dim3 updateThreads(8,3);
	updatePositionAndVelocity<<<updateBlocks, updateThreads>>>(d_accel_sum, d_hPos, d_hVel);
}


__global__ void fillAccelSum(vector3* accel_sum)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y;

	if (i >= NUMENTITIES) return;
	accel_sum[i][k] = 0;
}


__global__ void calculateAccels(vector3** accels, vector3* hPos, double* mass)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z;
	__shared__ vector3 distance[8][8];
	if (i >= NUMENTITIES || j >= NUMENTITIES) return;
	if (i==j)
	{
		accels[i][j][k] = 0;
	}
	else
	{
		distance[threadIdx.x][threadIdx.y][k] = hPos[i][k] - hPos[j][k];
		__syncthreads();
		double magnitude_sq = distance[threadIdx.x][threadIdx.y][0] * distance[threadIdx.x][threadIdx.y][0] + distance[threadIdx.x][threadIdx.y][1] * distance[threadIdx.x][threadIdx.y][1] + distance[threadIdx.x][threadIdx.y][2] * distance[threadIdx.x][threadIdx.y][2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
		accels[i][j][k] = accelmag*distance[threadIdx.x][threadIdx.y][k]/magnitude;
	}
}


__global__ void sumColumns(vector3** accels, vector3* accel_sum)
{
	int rowIndex = threadIdx.x;
	int colIndex = blockIdx.x;
	int dimension = blockIdx.y;
	__shared__ int offset;
	int blocksize = blockDim.x;
	int arraysize = NUMENTITIES;
	extern __shared__ double shArr[];
	shArr[rowIndex] = rowIndex < arraysize ? accels[colIndex][rowIndex][dimension] : 0;
	if (rowIndex == 0)
	{
		offset = blocksize;
	}
	__syncthreads();
	while (offset < arraysize)
	{
		shArr[rowIndex+blocksize] = rowIndex+blocksize < arraysize ? accels[colIndex][rowIndex+offset][dimension] : 0;
		__syncthreads();
		if (rowIndex == 0)
		{
			offset += blocksize;
		}
		double sum = shArr[2*rowIndex] + shArr[2*rowIndex+1];
		__syncthreads();
		shArr[rowIndex] = sum;
	}
	__syncthreads();
	for (int stride = 1; stride < blocksize; stride*=2)
	{
		int arrIdx = rowIndex*stride*2;
		if (arrIdx+stride < blocksize)
		{
			shArr[arrIdx] += shArr[arrIdx+stride];
		}
		__syncthreads();
	}
	if (rowIndex == 0)
	{
		accel_sum[colIndex][dimension] = shArr[0];
	}
}

__global__ void updatePositionAndVelocity(vector3* accel_sum, vector3* hPos, vector3* hVel)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = threadIdx.y;
	if (i >= NUMENTITIES) return;

	hVel[i][k] += accel_sum[i][k] * INTERVAL;
	hPos[i][k] += hVel[i][k] * INTERVAL;
}