#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#include<cassert>

int main(int argc, char* argv[]) {
    using thrust::host_vector;
    using thrust::device_vector;
    using thrust::transform;
    using thrust::plus;
    using thrust::raw_pointer_cast;

    cublasHandle_t handle;
    const int n = 1e8;
    const double alpha = 1.0;

    cublasCreate(&handle);

    // Initialize x and y on the host
    host_vector<double> h_x(n), h_y(n);
    for(int i=0; i<n; i++) {
        h_x[i] = i;
        h_y[i] = n-i;
    }

    // Copy x and y to device
    device_vector<double> x = h_x;
    device_vector<double> y = h_y;
    device_vector<double> z(n);

    // Compute sum of x and y
    cublasDcopy(handle, n, raw_pointer_cast(x.data()), 1, raw_pointer_cast(z.data()), 1);
    cublasDaxpy(handle, n, &alpha, raw_pointer_cast(y.data()), 1, raw_pointer_cast(z.data()), 1);

    // Copy z to host
    host_vector<double> h_z = z;

    // Assert that the sum is correct
    for(int i=0; i<n; i++) {
        assert(h_z[i] == n);
    }

    cublasDestroy(handle);

    return 0;
}