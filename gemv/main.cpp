// Build with nvcc -o gemv -acc -Minfo ../main.cpp

#include<cassert>
#include<vector>

int main(int argc, char* argv[]) {
    using std::vector;

    const int n = 10000;
    vector<double> x(n), y(n), A(n*n, 0.0);

    // Initialize x
    for(int i=0; i<n; i++) {
        x[i] = i;
    }

    // Initialize A to the identity matrix
    for(int i=0; i<n; i++) {
        A[i*(n+1)] = 1.0;
    }

    // Compute matrix-vector product
    #pragma acc parallel loop
    for(int r=0; r<n; r++) {
        y[r] = 0.0;
        for(int c=0; c<n; c++) {
            // Note that we have hard-coded row-major storage here
            y[r] += A[r*n+c] * x[c];
        }
    }

    // Assert that the product is correct
    for(int i=0; i<n; i++) {
        assert(y[i] == i);
    }

    return 0;
}
