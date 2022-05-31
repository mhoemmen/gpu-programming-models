#include<cassert>

#define THREADS_PER_BLOCK 512

// See https://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf
__global__ void dot( int n, double *a, double *b, double *c ) {
    __shared__ double temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n)
        temp[threadIdx.x] = a[index] * b[index];
    else
        temp[threadIdx.x] = 0.0;
    __syncthreads();
    if( 0 == threadIdx.x ) {
        double sum = 0;
        for( int i = 0; i < THREADS_PER_BLOCK; i++ )
            sum += temp[i];
        atomicAdd( c , sum );
    }
}

int main(int argc, char* argv[]) {
    const int n = 1e8;
    double* h_x = new double[n];
    double* h_y = new double[n];
    double h_sum;
    double *x, *y, *sum;

    cudaMalloc(&x, n*sizeof(double));
    cudaMalloc(&y, n*sizeof(double));
    cudaMalloc(&sum, sizeof(double));

    // Initialize x and y on the host
    for(int i=0; i<n; i++) {
        h_x[i] = i+1;
        h_y[i] = 1.0/h_x[i];
    }

    // Copy x and y to device
    cudaMemcpy(x, h_x, n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y, n*sizeof(double), cudaMemcpyHostToDevice);

    // Compute dot product
    dot<<< (n+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( n, x, y, sum );

    // Copy z to host
    cudaMemcpy(&h_sum, sum, sizeof(double), cudaMemcpyDeviceToHost);

    // Assert that the sum is correct
    assert(h_sum == n);

    delete[] h_x;
    delete[] h_y;
    cudaFree(x);
    cudaFree(y);
    cudaFree(sum);

    return 0;
}