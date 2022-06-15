#include<cassert>
#include<memory>
#include<vector>
#include<Kokkos_Core.hpp>

using Kokkos::ALL;
using Kokkos::LayoutStride;
using Kokkos::View;
using Kokkos::create_mirror_view;
using Kokkos::deep_copy;
using Kokkos::kokkos_free;
using Kokkos::kokkos_malloc;
using Kokkos::parallel_for;
using std::shared_ptr;
using std::vector;

#define infinity Kokkos::Experimental::infinity_v<double>
#define ORIGIN 0
#define DIRECTION 1
#define x 0
#define y 1
#define z 2
#define Ray View<const double[2][3], LayoutStride>

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
    KOKKOS_FUNCTION
    virtual double intersect(Ray r) const = 0;
};

// The plane x=0
// NOTE FOR MARK: This code crashes at runtime if I use
// class XPlane : public Shape {

class XPlane {
public:
    // It's polite to write "override" (after the close paren, before const),
    // since this method overrides a virtual base class function.
    KOKKOS_FUNCTION
    double intersect(Ray r) const; //override; // (Commented out because the inheritance has been commented out)
};

// The plane y=0
class YPlane : public Shape {
public:
    KOKKOS_FUNCTION
    double intersect(Ray r) const;
};

// The plane z=0
class ZPlane : public Shape {
public:
    KOKKOS_FUNCTION
    double intersect(Ray r) const;
};

// View can't store pointers, so we need to put a struct around it
// This is not a limitation of md_span
class ShapePtr {
public:
    ShapePtr() { shape_ = (XPlane*)kokkos_malloc<>(sizeof(XPlane)); }
    //~ShapePtr() { if(shape_) kokkos_free<>(shape_); }
    ShapePtr(const ShapePtr&) = delete;
    ShapePtr& operator=(const ShapePtr&) = delete;

    KOKKOS_FUNCTION void make();

    KOKKOS_FUNCTION
    double intersect(Ray r) const { return shape_->intersect(r); }

private:
    XPlane* shape_;
};

class Geometry {
public:
    Geometry(int nshapes) :
        shapes_("shapes", nshapes) {}

    void addXPlane(int i);

    // __host__ __device__ with cuda, null string without
    KOKKOS_FUNCTION
    double intersect(Ray r) const;
private:
    // Recall that we can't store a vector of an abstract class, 
    // so we store a vector of pointers instead
    View<ShapePtr*> shapes_;
};

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    { // View destructors must be invoked before calling finalize, 
      // so it is common to place code in a block.
        const int NRAYS = 3, NSHAPES = 3;
        Geometry g(NSHAPES);
        // Why store the rays in this way?
        // First, because optimal storage is different on host vs device.
        // Second, because storing arrays of objects that contain more 
        // arrays on the device is _complicated_.
        View<double*[2][3]> rays("rays", NRAYS);
        View<double*> distances("distances", NRAYS);

        // Set the ray values on the host
        auto h_rays = create_mirror_view(rays);
        h_rays(0,ORIGIN,0) = 1.0; h_rays(0,ORIGIN,1) = 1.0; h_rays(0,ORIGIN,2) = 1.0;
        h_rays(1,ORIGIN,0) = 2.0; h_rays(1,ORIGIN,1) = 1.0; h_rays(1,ORIGIN,2) = 1.0;
        h_rays(2,ORIGIN,0) = 1.0; h_rays(2,ORIGIN,1) = 1.0; h_rays(2,ORIGIN,2) = 0.0;
        h_rays(0,DIRECTION,0) = -1.0; h_rays(0,DIRECTION,1) = 0.0; h_rays(0,DIRECTION,2) = 0.0;
        h_rays(1,DIRECTION,0) = 1.0; h_rays(1,DIRECTION,1) = 1.0; h_rays(1,DIRECTION,2) = 1.0;
        h_rays(2,DIRECTION,0) = -1.0; h_rays(2,DIRECTION,1) = -1.0; h_rays(2,DIRECTION,2) = -1.0;

        // Copy the rays to device
        deep_copy(rays, h_rays);

        // Create the geometry
        g.addXPlane(0);
        g.addXPlane(1);
        g.addXPlane(2);

        // Compute the distance to intersection
        parallel_for("intersect", NRAYS, KOKKOS_LAMBDA(int i) {
            auto ray = subview(rays, i, ALL(), ALL());
            distances(i) = g.intersect(ray);
        });

        // Copy the distances back to the host
        auto h_distances = create_mirror_view(distances);
        deep_copy(h_distances, distances);

        // Check the distances
        assert(h_distances[0] == 1.0);
        assert(h_distances[1] == infinity);
        assert(h_distances[2] == 0.0);

    }
    Kokkos::finalize();

    return 0;
}

void ShapePtr::make() {
    // We have to invoke placement new on the device or else the vtable gets screwy
    // See https://github.com/kokkos/kokkos/wiki/Kokkos-and-Virtual-Functions
    new((XPlane*)shape_) XPlane();
}

KOKKOS_FUNCTION
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

double XPlane::intersect(Ray r) const {
    return intersect_helper(r(ORIGIN,0), r(DIRECTION,0));
}

double YPlane::intersect(Ray r) const {
    return intersect_helper(r(ORIGIN,1), r(DIRECTION,1));
}

double ZPlane::intersect(Ray r) const {
    return intersect_helper(r(ORIGIN,2), r(DIRECTION,2));
}

void Geometry::addXPlane(int i) {
    // We have to invoke placement new on the device or else the vtable gets screwy
    // See https://github.com/kokkos/kokkos/wiki/Kokkos-and-Virtual-Functions
    parallel_for("addXPlane", 1, KOKKOS_CLASS_LAMBDA(int) {
        shapes_(i).make();
    });
}
    
double Geometry::intersect(Ray r) const {
    using Kokkos::Experimental::fmin;
    
    double min_dist = infinity;
    for(int i=0; i<shapes_.extent(0); i++) {
        min_dist = fmin(min_dist, shapes_(i).intersect(r));
    }

    return min_dist;
}
