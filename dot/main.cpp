#include<algorithm>
#include<cassert>
#include<execution>
#include<functional>
#include<numeric>
#include<vector>

int main(int argc, char* argv[]) {
    using std::execution::par_unseq;
    using std::vector;

    const int n = 100'000'000;
    vector<double> x(n), y(n);

    // Initialize x and y
    for(int i=0; i<n; i++) {
        x[i] = i+1;
        y[i] = 1.0/x[i];
    }

    // Compute dot product
    double sum = std::transform_reduce(par_unseq,
                          x.begin(), x.end(),
                          y.begin(), 0.0, std::plus<double>(),
                          [](const double& xi, const double& yi){
                              return xi * yi;
                          });

    // Assert that the sum is correct
    assert(sum == n);

    return 0;
}
