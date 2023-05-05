void compute();
__global__ void mapValuestoAccels(vector3* values, vector3** accels);
__global__ void fillAccelSum(vector3* accel_sum);
__global__ void calculateAccels(vector3** accels, vector3* hPos, double* mass);
__global__ void sumColumns(vector3** accels, vector3* accel_sum);
__global__ void updatePositionAndVelocity(vector3* accel_sum, vector3* hPos, vector3* hVel);