#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cassert>

void axpy(int n, double* x, double* y, double* z, sycl::nd_item<3> item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n) {
        z[i] = x[i] + y[i];
    }
}

int main(int argc, char *argv[]) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    const int n = 1e8;
    double* h_x = new double[n];
    double* h_y = new double[n];
    double* h_z = new double[n];
    double *x, *y, *z;

    x = sycl::malloc_device<double>(n, q_ct1);
    y = sycl::malloc_device<double>(n, q_ct1);
    z = sycl::malloc_device<double>(n, q_ct1);

    // Initialize x and y on the host
    for(int i=0; i<n; i++) {
        h_x[i] = i;
        h_y[i] = n-i;
    }

    // Copy x and y to device
    q_ct1.memcpy(x, h_x, n * sizeof(double));
    q_ct1.memcpy(y, h_y, n * sizeof(double)).wait();

    // Compute sum of x and y
    q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (n + 255) / 256) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                           axpy(n, x, y, z, item_ct1);
                       });

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

    return 0;
}