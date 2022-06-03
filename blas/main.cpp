#include<cassert>
#include<vector>

extern "C" {
  void dcopy_(const int* n, const double* x, const int* incx, double* y, const int* incy);

  void daxpy_(const int* n, const double* alpha, 
              const double* x, const int* incx,
              double* y, const int* incy);
} // end extern

int main(int argc, char* argv[]) {
    using std::vector;

    const int n = 100'000'000;
    const int ONE = 1;
    const double alpha = 1.0;
    vector<double> x(n), y(n), z(n);

    // Initialize x and y
    for(int i=0; i<n; i++) {
        x[i] = i;
        y[i] = n-i;
    }

    // Compute sum of x and y
    dcopy_(&n, x.data(), &ONE, z.data(), &ONE);
    daxpy_(&n, &alpha, y.data(), &ONE, z.data(), &ONE);
    for(int i=0; i<n; i++) {
        z[i] = x[i] + y[i];
    }

    // Assert that the sum is correct
    for(int i=0; i<n; i++) {
        assert(z[i] == n);
    }

    return 0;
}