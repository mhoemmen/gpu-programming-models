#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#include <cassert>

int main(int argc, char* argv[]) {
    using thrust::host_vector;
    using thrust::device_vector;
    using thrust::transform;
    using thrust::plus;
    using thrust::raw_pointer_cast;

    sycl::queue *handle;
    const int n = 1e8;
    const double alpha = 1.0;

    handle = &dpct::get_default_queue();

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
    (/*
    DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
     oneapi::mkl::blas::column_major::copy(*handle, n,
                                           raw_pointer_cast(x.data()), 1,
                                           raw_pointer_cast(z.data()), 1),
     0);
    (/*
    DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
     oneapi::mkl::blas::column_major::axpy(*handle, n, alpha,
                                           raw_pointer_cast(y.data()), 1,
                                           raw_pointer_cast(z.data()), 1),
     0);

    // Copy z to host
    host_vector<double> h_z = z;

    // Assert that the sum is correct
    for(int i=0; i<n; i++) {
        assert(h_z[i] == n);
    }

    handle = nullptr;

    return 0;
}