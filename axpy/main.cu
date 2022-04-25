#include<cassert>

__global__ 
void axpy(int n, double* x, double* y, double* z) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        z[i] = x[i] + y[i];
    }
}

int main(int argc, char* argv[]) {
    const int n = 1e8;
    double* h_x = new double[n];
    double* h_y = new double[n];
    double* h_z = new double[n];
    double *x, *y, *z;

    cudaMalloc(&x, n*sizeof(double));
    cudaMalloc(&y, n*sizeof(double));
    cudaMalloc(&z, n*sizeof(double));

    // Initialize x and y on the host
    for(int i=0; i<n; i++) {
        h_x[i] = i;
        h_y[i] = n-i;
    }

    // Copy x and y to device
    cudaMemcpy(x, h_x, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y, n*sizeof(double), cudaMemcpyHostToDevice);

    // Compute sum of x and y
    axpy<<<(n+255)/256, 256>>>(n, x, y, z);

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

    return 0;
}