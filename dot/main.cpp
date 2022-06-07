// Build with nvcc -o dot -acc -Minfo ../main.cpp

#include<cassert>
#include<vector>

int main(int argc, char* argv[]) {
    using std::vector;

    const int n = 100'000'000;
    vector<double> x(n), y(n);

    // Initialize x and y
    for(int i=0; i<n; i++) {
        x[i] = i+1;
        y[i] = 1.0/x[i];
    }

    // Compute dot product
    double sum = 0.0;
    #pragma acc parallel loop
    for(int i=0; i<n; i++) {
        sum += x[i] * y[i];
    }

    // Assert that the sum is correct
    assert(sum == n);

    return 0;
}