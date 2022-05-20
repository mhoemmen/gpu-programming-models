#include<cassert>
#include "chai/ManagedArray.hpp"
#include "RAJA/RAJA.hpp"

int main(int argc, char* argv[]) {
    using chai::ManagedArray;
    using RAJA::forall;
    using RAJA::cuda_exec;
    using RAJA::RangeSegment;

    const int n = 1e8;
    const int CUDA_BLOCK_SIZE = 256;
    chai::ManagedArray<double> x(n), y(n), z(n);

    // Initialize x and y
    for(int i=0; i<n; i++) {
        x[i] = i;
        y[i] = n-i;
    }

    // Compute sum of x and y
    forall<cuda_exec<CUDA_BLOCK_SIZE>>(RangeSegment(0, n), 
        [=] RAJA_DEVICE (int i) { 
        z[i] = x[i] + y[i]; 
    });    

    // Assert that the sum is correct
    for(int i=0; i<n; i++) {
        assert(z[i] == n);
    }

    return 0;
}