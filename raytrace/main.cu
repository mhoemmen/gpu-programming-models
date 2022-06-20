#include<cassert>
#include<vector>

using std::vector;

constexpr auto infinity = std::numeric_limits<double>::infinity();

class Shape {
public:
    // C++ Core Guidelines C.126: An abstract class typically doesn't need a user-written constructor
    Shape() = default;
    // C++ Core Guidelines C.21: If you define or =delete any copy, move, or destructor function, define or =delete them all
    // (For an abstract base class, you probably meant to delete them all.)
    Shape(const Shape&) = delete;
    Shape& operator=(const Shape&) = delete;
    // C++ Core Guidelines C.127: A class with a virtual function should have a virtual or protected destructor
    virtual ~Shape() = default;

    // Returns the distance to the intersection point
    // infinity if there is no intersection
    __device__
    virtual double intersect(double* ray, int stride) const = 0;
};

// The plane x=0
class XPlane : public Shape {
public:
    ~XPlane() override = default;

    // It's polite to write "override" (after the close paren, before const),
    // since this method overrides a virtual base class function.
    __device__
    double intersect(double* ray, int stride) const override;
};

// The plane y=0
class YPlane : public Shape {
public:
    ~YPlane() override = default;

    __device__
    double intersect(double* ray, int stride) const override;
};

// The plane z=0
class ZPlane : public Shape {
public:
    ~ZPlane() override = default;
    
    __device__
    double intersect(double* ray, int stride ) const override;
};

class Geometry {
public:
    void addXPlane();
    void addYPlane();
    void addZPlane();
    void fillComplete();

    __device__
    double intersect(double* ray, int stride) const;
private:
    // Recall that we can't store a vector of an abstract class, 
    // so we store a vector of pointers instead
    Shape** shapes_;
    vector<Shape*> h_shapes_;
    int nshapes_;
};

// We have to invoke placement new on the device or else the vtable gets screwy
// See https://github.com/kokkos/kokkos/wiki/Kokkos-and-Virtual-Functions
__global__ void makeXPlane(Shape* shapeptr) { 
    new((XPlane*)shapeptr) XPlane(); 
}

__global__ void makeYPlane(Shape* shapeptr) { 
    new((YPlane*)shapeptr) YPlane(); 
}

__global__ void makeZPlane(Shape* shapeptr) { 
    new((ZPlane*)shapeptr) ZPlane(); 
}

__global__
void compute_intersections(double* rays, int stride, double* distances, Geometry g) {
    int i = threadIdx.x;
    double* ray = rays + i;
    distances[i] = g.intersect(ray, stride);
}

int main(int argc, char* argv[]) {
    const int NRAYS = 3;
    Geometry g;

    // Create host arrays
    vector<double> h_rays {1.0, 2.0, 1.0, -1.0, 1.0, -1.0, 
                           1.0, 1.0, 1.0,  0.0, 1.0, -1.0, 
                           1.0, 1.0, 0.0, 0.0, 1.0, -1.0};
    vector<double> h_distances(NRAYS);

    // Create device arrays
    double *rays, *distances;
    cudaMalloc(&rays, NRAYS*2*3*sizeof(double));
    cudaMalloc(&distances, NRAYS*sizeof(double));

    // Copy the rays to device
    cudaMemcpy(rays, h_rays.data(), NRAYS*2*3*sizeof(double), cudaMemcpyHostToDevice);

    // Create the geometry
    g.addXPlane();
    g.addYPlane();
    g.addZPlane();
    g.fillComplete();

    // Compute the distance to intersection
    compute_intersections<<<1,NRAYS>>>(rays, NRAYS, distances, g);

    // Copy the distances back to the host
    cudaMemcpy(h_distances.data(), distances, NRAYS*sizeof(double), cudaMemcpyDeviceToHost);

    // Check the distances
    assert(h_distances[0] == 1.0);
    assert(h_distances[1] == infinity);
    assert(h_distances[2] == 0.0);

    cudaFree(rays);
    cudaFree(distances);

    return 0;
}

__device__
double intersect_helper(double origin_i, double direction_i) {
    if(origin_i == 0.0) {
        return 0.0;
    } else if(direction_i == 0.0) {
        return infinity;
    }

    double t = -origin_i / direction_i;
    if(t <= 0.0) {
        return infinity;
    }
    return t;
}

__device__
double XPlane::intersect(double* ray, int stride) const {
    return intersect_helper(ray[0], ray[3]);
}

__device__
double YPlane::intersect(double* ray, int stride) const {
    return intersect_helper(ray[2*stride], ray[3*stride]);
}

__device__
double ZPlane::intersect(double* ray, int stride) const {
    return intersect_helper(ray[4*stride], ray[5*stride]);
}

void Geometry::addXPlane() {
    Shape* shapeptr;
    cudaMalloc(&shapeptr, sizeof(XPlane));
    makeXPlane<<<1,1>>>(shapeptr);
    h_shapes_.push_back(shapeptr);
}

void Geometry::addYPlane() {
    Shape* shapeptr;
    cudaMalloc(&shapeptr, sizeof(YPlane));
    makeYPlane<<<1,1>>>(shapeptr);
    h_shapes_.push_back(shapeptr);
}

void Geometry::addZPlane() {
    Shape* shapeptr;
    cudaMalloc(&shapeptr, sizeof(ZPlane));
    makeZPlane<<<1,1>>>(shapeptr);
    h_shapes_.push_back(shapeptr);
}

void Geometry::fillComplete() {
    nshapes_ = h_shapes_.size();
    cudaMalloc(&shapes_, nshapes_*sizeof(Shape*));
    cudaMemcpy(shapes_, h_shapes_.data(), nshapes_*sizeof(Shape*), cudaMemcpyHostToDevice);
    h_shapes_.clear();
}
    
__device__
double Geometry::intersect(double* ray, int stride) const {
    double min_dist = infinity;
    for(int i=0; i<nshapes_; i++) {
        min_dist = fmin(min_dist, shapes_[i]->intersect(ray, stride));
    }

    return min_dist;
}
