#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#include<cassert>

int main(int argc, char* argv[]) {
    cublasHandle_t handle;
    const int n = 1e8;
    const double alpha = 1.0;
    double* h_x = new double[n];
    double* h_y = new double[n];
    double* h_z = new double[n];
    double *x, *y, *z;

    cudaMalloc(&x, n*sizeof(double));
    cudaMalloc(&y, n*sizeof(double));
    cudaMalloc(&z, n*sizeof(double));

    cublasCreate(&handle);

    // Initialize x and y on the host
    for(int i=0; i<n; i++) {
        h_x[i] = i;
        h_y[i] = n-i;
    }

    // Copy x and y to device
    cudaMemcpy(x, h_x, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y, n*sizeof(double), cudaMemcpyHostToDevice);

    // Compute sum of x and y
    cublasDcopy(handle, n, x, 1, z, 1);
    cublasDaxpy(handle, n, &alpha, y, 1, z, 1);

    // Copy z to host
    cudaMemcpy(h_z, z, n*sizeof(double), cudaMemcpyDeviceToHost);

    // Assert that the sum is correct
    for(int i=0; i<n; i++) {
        assert(h_z[i] == n);
    }

    delete[] h_x, h_y, h_z;
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    cublasDestroy(handle);

    return 0;
}