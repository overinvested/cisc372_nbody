#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "parallel_compute.h"


extern vector3 *d_values, **d_accels, **d_hPos, **d_hVel, *d_accel_sum;
extern double *d_mass;


//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	// int i,j,k;
	// vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	// vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	
	
	int numBlocks = 1;
	dim3 threadsPerBlock(NUMENTITIES, NUMENTITIES, 3);
	int sharedMemorySize = 2*NUMENTITIES*sizeof(double);
	mapValuestoAccels<<<numBlocks, threadsPerBlock>>>(d_values, d_accels);
	fillAccelSum<<<numBlocks, threadsPerBlock>>>(d_accel_sum);
	calculateAccels<<<numBlocks, threadsPerBlock>>>(d_accels, d_hPos, d_mass);
	sumColumns<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_accels, d_accel_sum, NUMENTITIES);
	updatePositionAndVelocity<<<numBlocks, threadsPerBlock>>>(d_accel_sum, d_hPos, d_hVel);


	// for (i=0;i<NUMENTITIES;i++)
	// 	accels[i]=&values[i*NUMENTITIES];
	// //first compute the pairwise accelerations.  Effect is on the first argument.
	// for (i=0;i<NUMENTITIES;i++){
	// 	for (j=0;j<NUMENTITIES;j++){
	// 		if (i==j) {
	// 			FILL_VECTOR(accels[i][j],0,0,0);
	// 		}
	// 		else{
	// 			vector3 distance;
	// 			for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
	// 			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
	// 			double magnitude=sqrt(magnitude_sq);
	// 			double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
	// 			FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	// 		}
	// 	}
	// }
	// //sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	// for (i=0;i<NUMENTITIES;i++){
	// 	vector3 accel_sum={0,0,0};
	// 	for (j=0;j<NUMENTITIES;j++){
	// 		for (k=0;k<3;k++)
	// 			accel_sum[k]+=accels[i][j][k];
	// 	}
	// 	//compute the new velocity based on the acceleration and time interval
	// 	//compute the new position based on the velocity and time interval
	// 	for (k=0;k<3;k++){
	// 		hVel[i][k]+=accel_sum[k]*INTERVAL;
	// 		hPos[i][k]=hVel[i][k]*INTERVAL;
	// 	}
	// }
	// free(accels);
	// free(values);
}


__global__ void mapValuestoAccels(vector3* values, vector3** accels)
{
	int i = threadIdx.x;

	accels[i] = &values[i*NUMENTITIES];
}


__global__ void fillAccelSum(vector3* accel_sum)
{
	int i = threadIdx.x;
	FILL_VECTOR(accel_sum[i],0,0,0);
	// *accel_sum[i] = {0,0,0};
}


__global__ void calculateAccels(vector3** accels, vector3** hPos, double* mass)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	if (i==j)
	{
		FILL_VECTOR(accels[i][j],0,0,0);
	}
	else
	{
		vector3 distance;
		for (int k = 0; k < 3; k++)
		{
			distance[k] = hPos[i][k] - hPos[j][k];
		}
		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
		FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	}
}


__global__ void sumColumns(vector3** accels, vector3* accel_sum, int blocksize)
{
	int rowIndex = threadIdx.x;
	int colIndex = threadIdx.y;
	int dimension = threadIdx.z;
	__shared__ int offset;
	int arraysize = blocksize*2;
	extern __shared__ vector3 shArr[];
	shArr[rowIndex][dimension] = rowIndex < arraysize ? accels[rowIndex][colIndex][dimension] : 0;
	if (rowIndex == 0)
	{
		offset = blocksize;
	}
	__syncthreads();
	while (offset < arraysize)
	{
		shArr[rowIndex+blocksize][dimension] = rowIndex+blocksize < arraysize ? accels[rowIndex+offset][colIndex][dimension] : 0;
		__syncthreads();
		if (rowIndex == 0)
		{
			offset += blocksize;
		}
		double sum = shArr[2*rowIndex][dimension] + shArr[2*rowIndex+1][dimension];
		__syncthreads();
		shArr[rowIndex][dimension] = sum;
	}
	__syncthreads();
	for (int stride = 1; stride < blocksize; stride<<=1)
	{
		int arrIdx = rowIndex*stride*2;
		if (arrIdx+stride < blocksize)
		{
			shArr[arrIdx][dimension] += shArr[arrIdx+stride][dimension];
		}
		__syncthreads();
	}
	if (rowIndex == 0)
	{
		accel_sum[colIndex][dimension] = shArr[0][dimension];
	}
}

__global__ void updatePositionAndVelocity(vector3* accel_sum, vector3** hPos, vector3** hVel)
{
	int i = threadIdx.x;
	// int j = threadIdx.y;
	int k = threadIdx.z;

	*hVel[i][k] += *accel_sum[k] * INTERVAL;
	*hPos[i][k] = *hVel[i][k] * INTERVAL;
}