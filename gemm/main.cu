// Perform a block multiplication of two matrices
// For simplicity, we assume the number of blocks evenly divide the number of rows
// We use block column major storage, and column major within blocks

#include<cassert>
#include<random>
#include<vector>

constexpr auto n = 20;  // Number of rows
constexpr auto nb = 1;   // Number of blocks in a given direction
constexpr auto bs = n/nb; // Block size

extern "C" {
  void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
              const double* alpha, const double* A, const int* lda,
              const double* B, const int* ldb,
              const double* beta, double* C, const int* ldc);
} // end extern

// Returns the starting index for a given block
int getStartingIndex(int blockrow, int blockcol) {
    return (blockrow + blockcol*nb)*bs*bs;
}

int getIndex(int row, int col) {
    int br = row / bs;
    int r = row % bs;
    int bc = col / bs;
    int c = col % bs;
    return nb*bs*bs*bc + bs*bs*br + bs*c + r;
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
    // Because we're using a strange storage format, this takes several BLAS calls
    constexpr auto trans = 'N';
    constexpr auto alpha = 1.0;
    constexpr auto beta = 1.0;
    for(int rb=0; rb<nb; rb++) {
        for(int cb=0; cb<nb; cb++) {
            double* cptr = Cgs.data() + getStartingIndex(rb,cb);
            // Zero out the C block
            for(int i=0; i<bs; i++) {
                cptr[i] = 0.0;
            }
            for(int i=0; i<nb; i++) {
                double* aptr = A.data() + getStartingIndex(rb,i);
                double* bptr = B.data() + getStartingIndex(i,cb);
                dgemm_(&trans, &trans, &bs, &bs, &bs, &alpha, aptr, &bs, bptr, &bs, &beta, cptr, &bs);
            }
        }
    }

    // Assert that the product is correct
    for(int i=0; i<n*n; i++) {
        assert(C[i] == Cgs[i]);
    }

    return 0;
}
