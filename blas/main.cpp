#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#include <cassert>

int main(int argc, char *argv[]) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    sycl::queue *handle;
    const int n = 1e8;
    const double alpha = 1.0;
    double* h_x = new double[n];
    double* h_y = new double[n];
    double* h_z = new double[n];
    double *x, *y, *z;

    x = sycl::malloc_device<double>(n, q_ct1);
    y = sycl::malloc_device<double>(n, q_ct1);
    z = sycl::malloc_device<double>(n, q_ct1);

    handle = &q_ct1;

    // Initialize x and y on the host
    for(int i=0; i<n; i++) {
        h_x[i] = i;
        h_y[i] = n-i;
    }

    // Copy x and y to device
    q_ct1.memcpy(x, h_x, n * sizeof(double));
    q_ct1.memcpy(y, h_y, n * sizeof(double)).wait();

    // Compute sum of x and y
    oneapi::mkl::blas::column_major::copy(*handle, n, x, 1, z, 1);
    oneapi::mkl::blas::column_major::axpy(*handle, n, alpha, y, 1, z, 1);

    // Copy z to host
    q_ct1.memcpy(h_z, z, n * sizeof(double)).wait();

    // Assert that the sum is correct
    for(int i=0; i<n; i++) {
        assert(h_z[i] == n);
    }

    delete[] h_x, h_y, h_z;
    sycl::free(x, q_ct1);
    sycl::free(y, q_ct1);
    sycl::free(z, q_ct1);

    handle = nullptr;

    return 0;
}