#include<cassert>
#include<Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    typedef Kokkos::TeamPolicy<>               team_policy;
    typedef Kokkos::TeamPolicy<>::member_type  member_type;

    using Kokkos::AUTO;
    using Kokkos::View;
    using Kokkos::create_mirror_view;
    using Kokkos::deep_copy;
    using Kokkos::parallel_for;

    Kokkos::initialize(argc, argv);
    { // View destructors must be invoked before calling finalize, 
      // so it is common to place code in a block.
        const int n = 10000;
        View<double*> x("x", n), y("y", n);
        View<double**> A("A", n, n);

        auto h_x = create_mirror_view(x);
        auto h_y = create_mirror_view(y);
        auto h_A = create_mirror_view(A);

        // Initialize x on host
        for(int i=0; i<n; i++) {
            h_x[i] = i;
        }

        // Initialize A to the identity matrix on host
        for(int i=0; i<n; i++) {
            h_A(i,i) = 1.0;
        }

        // Copy data to device
        deep_copy(x, h_x);
        deep_copy(A, h_A);

        // Compute matrix-vector product
        parallel_for("gemv", team_policy(n, AUTO), KOKKOS_LAMBDA(const member_type &teamMember) {
            int r = teamMember.league_rank();
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange( teamMember, n ), [&] ( const int c, double &localSum ) {
                localSum += A(r,c) * x(c);
            }, y(r));
        });

        // Copy data to host
        deep_copy(h_y, y);

        // Assert that the product is correct
        for(int i=0; i<n; i++) {
            assert(h_y[i] == i);
        }
    }
    Kokkos::finalize();

    return 0;
}