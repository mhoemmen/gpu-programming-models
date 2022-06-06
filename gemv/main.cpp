#include<cassert>
#include<execution>
#include<ranges>
#include<vector>

int main(int argc, char* argv[]) {
    using std::execution::par_unseq;
    using std::for_each;
    using std::ranges::begin;
    using std::ranges::end;
    using std::views::iota;
    using std::vector;

    const int n = 1000;
    vector<double> x(n), y(n), A(n*n, 0.0);

    // Initialize x
    for(int i=0; i<n; i++) {
        x[i] = i;
    }

    // Initialize A to the identity matrix
    for(int i=0; i<n; i++) {
        A[i*(n+1)] = 1.0;
    }

    // Generate a range of indices
    auto rows = iota(0, n);

    // Compute matrix-vector product
    for_each(par_unseq, begin(rows), end(rows), [=,&y](int r) {
        y[r] = 0.0;
        for(int c=0; c<n; c++) {
            // Note that we have hard-coded row-major storage here
            y[r] += A[r*n+c] * x[c];
        }
    });

    // Assert that the product is correct
    for(int i=0; i<n; i++) {
        assert(y[i] == i);
    }

    return 0;
}