#include<algorithm>
#include<cassert>
#include<execution>
#include<functional>
#include<vector>

int main(int argc, char* argv[]) {
    using std::execution::par_unseq;
    using std::vector;

    const int n = 100'000'000;
    vector<double> x(n), y(n), z(n);

    // Initialize x and y
    for(int i=0; i<n; i++) {
        x[i] = i;
        y[i] = n-i;
    }

    // Compute sum of x and y
    std::transform(par_unseq, x.cbegin(), x.cend(), y.cbegin(),
                   z.begin(), std::plus<>{});

    // Assert that the sum is correct
    for(int i=0; i<n; i++) {
        assert(z[i] == n);
    }

    return 0;
}
