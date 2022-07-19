#include<cassert>
#include<random>
#include<vector>

constexpr auto n = 2000;

extern "C" {
  void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
              const double* alpha, const double* A, const int* lda,
              const double* B, const int* ldb,
              const double* beta, double* C, const int* ldc);
} // end extern

int getIndex(int row, int col) {
    // Note that we have hard coded column major storage here
    return row + col*n;
}

int main(int argc, char* argv[]) {
    using std::vector;
    vector<double> A(n*n), B(n*n), C(n*n), Cgs(n*n);

    // Set up random number generator
    // Using integers for simpler testing
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,100);

    // Initialize A and B
    for(int i=0; i<n*n; i++) {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }

    // Compute matrix-vector product
    for(int r=0; r<n; r++) {
        for(int c=0; c<n; c++) {
            int cind = getIndex(r, c);
            C[cind] = 0.0;
            for(int i=0; i<n; i++) {
                int aind = getIndex(r, i);
                int bind = getIndex(i, c);
                C[cind] += A[aind]*B[bind];
            }
        }
    }

    // Compute gold standard using BLAS
    constexpr auto trans = 'N';
    constexpr auto alpha = 1.0;
    constexpr auto beta = 0.0;
    dgemm_(&trans, &trans, &n, &n, &n, &alpha, A.data(), &n, B.data(), &n, &beta, Cgs.data(), &n);

    // Assert that the product is correct
    for(int i=0; i<n*n; i++) {
        assert(C[i] == Cgs[i]);
    }

    return 0;
}
