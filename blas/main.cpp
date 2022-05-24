#include<cassert>
#include<Kokkos_Core.hpp>
#include<KokkosBlas1_update.hpp>

int main(int argc, char* argv[]) {
    using Kokkos::View;
    using Kokkos::create_mirror_view;
    using Kokkos::deep_copy;

    Kokkos::initialize(argc, argv);
    { // View destructors must be invoked before calling finalize, 
      // so it is common to place code in a block.
        const int n = 100'000'000;
        View<double*> x("x", n), y("y", n), z("z", n);

        auto h_x = create_mirror_view(x);
        auto h_y = create_mirror_view(y);
        auto h_z = create_mirror_view(z);

        // Initialize x and y on host
        for(int i=0; i<n; i++) {
            h_x[i] = i;
            h_y[i] = n-i;
        }

        // Copy data to device
        deep_copy(x, h_x);
        deep_copy(y, h_y);

        // Compute sum of x and y
        KokkosBlas::update(1.0, x, 1.0, y, 0.0, z);

        // Copy data to host
        deep_copy(h_z, z);

        // Assert that the sum is correct
        for(int i=0; i<n; i++) {
            assert(h_z[i] == n);
        }
    }
    Kokkos::finalize();

    return 0;
}