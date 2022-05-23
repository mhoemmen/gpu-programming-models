#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include<cassert>

int main(int argc, char* argv[]) {
    using thrust::host_vector;
    using thrust::device_vector;
    using thrust::transform;
    using thrust::plus;

    const int n = 1e8;
    host_vector<double> h_x(n), h_y(n);

    // Initialize x and y on the host
    for(int i=0; i<n; i++) {
        h_x[i] = i;
        h_y[i] = n-i;
    }

    // Copy x and y to device
    device_vector<double> x = h_x;
    device_vector<double> y = h_y;
    device_vector<double> z(n);

    // Compute sum of x and y
    transform(x.begin(), x.end(), y.begin(), z.begin(), plus<double>());

    // Copy z to host
    host_vector<double> h_z = z;

    // Assert that the sum is correct
    for(int i=0; i<n; i++) {
        assert(h_z[i] == n);
    }

    return 0;
}