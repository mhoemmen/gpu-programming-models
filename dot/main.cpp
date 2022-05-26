#include<cassert>
#include<Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    using Kokkos::View;
    using Kokkos::create_mirror_view;
    using Kokkos::deep_copy;
    using Kokkos::parallel_reduce;

    Kokkos::initialize(argc, argv);
    { // View destructors must be invoked before calling finalize, 
      // so it is common to place code in a block.
        const int n = 100'000'000;
        View<double*> x("x", n), y("y", n);

        auto h_x = create_mirror_view(x);
        auto h_y = create_mirror_view(y);

        // Initialize x and y on host
        for(int i=0; i<n; i++) {
            h_x[i] = i+1;
            h_y[i] = 1.0/h_x[i];
        }

        // Copy data to device
        deep_copy(x, h_x);
        deep_copy(y, h_y);

        // Compute dot product
        double sum = 0.0;
        parallel_reduce("dot", n, KOKKOS_LAMBDA (const int& i, double& lsum ) {
            lsum += x[i] * y[i];
        }, sum);

        // Assert that the sum is correct
        assert(sum == n);
    }
    Kokkos::finalize();

    return 0;
}