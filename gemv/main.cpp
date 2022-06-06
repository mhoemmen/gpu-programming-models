#include<algorithm>
#include<cassert>
#include<execution>
#include<functional>
#include<numeric>
#include<vector>

int main(int argc, char* argv[]) {
    using std::execution::par_unseq;
    using std::for_each;
    using std::transform_reduce;
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
    vector<int> rows(n);
    for(int i=0; i<n; i++) {
        rows[i] = i;
    }

    // Compute matrix-vector product
    for_each(par_unseq, rows.begin(), rows.end(), [=,&y](int r) {
        // Note that we have hard-coded row-major storage here
        y[r] = transform_reduce(par_unseq, x.begin(), x.end(), 
            A.cbegin() + r*n, 0.0, std::plus<double>{}, std::multiplies<double>{});;
    });

    // Assert that the product is correct
    for(int i=0; i<n; i++) {
        assert(y[i] == i);
    }

    return 0;
}
