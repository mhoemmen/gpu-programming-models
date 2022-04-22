#include<cassert>
#include<vector>

int main(int argc, char* argv[]) {
    using std::vector;

    const int n = 1e8;
    vector<double> x(n), y(n), z(n);

    // Initialize x and y
    for(int i=0; i<n; i++) {
        x[i] = i;
        y[i] = n-i;
    }

    // Compute sum of x and y
    for(int i=0; i<n; i++) {
        z[i] = x[i] + y[i];
    }

    // Assert that the sum is correct
    for(int i=0; i<n; i++) {
        assert(z[i] == n);
    }

    return 0;
}