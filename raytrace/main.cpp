// Build with nvcc -o ray -acc -Minfo ../main.cpp

#include<algorithm>
#include<cassert>
#include<limits>
#include<memory>
#include<vector>

using std::shared_ptr;
using std::vector;

#define INFINITY std::numeric_limits<double>::infinity()

struct Ray {
    Ray(double org[3], double dir[3]);
    double origin[3];
    double direction[3];
};

class Shape {
public:
    // Returns the distance to the intersection point
    // Infinity if there is no intersection
    virtual double intersect(const Ray& r) const = 0;
};

// The plane x=0
class XPlane : public Shape {
public:
    double intersect(const Ray& r) const;
};

// The plane y=0
class YPlane : public Shape {
public:
    double intersect(const Ray& r) const;
};

// The plane z=0
class ZPlane : public Shape {
public:
    double intersect(const Ray& r) const;
};

class Geometry {
public:
    void add(shared_ptr<Shape> s);
    double intersect(const Ray& r) const;
private:
    // Recall that we can't store a vector of an abstract class, 
    // so we store a vector of pointers instead
    vector<shared_ptr<Shape>> shapes_;
};

int main(int argc, char* argv[]) {
    Geometry g;
    std::vector<Ray> rays;
    std::vector<double> distances;

    double p1[3] = {1.0, 1.0, 1.0};
    double p2[3] = {2.0, 1.0, 1.0};
    double p3[3] = {1.0, 1.0, 0.0};
    double d1[3] = {-1.0, 0.0, 0.0};
    double d2[3] = {1.0, 1.0, 1.0};
    double d3[3] = {-1.0, -1.0, -1.0};

    shared_ptr<Shape> xp = std::make_shared<XPlane>();
    shared_ptr<Shape> yp = std::make_shared<YPlane>();
    shared_ptr<Shape> zp = std::make_shared<ZPlane>();

    g.add(xp);
    g.add(yp);
    g.add(zp);

    rays.push_back(Ray(p1,d1));
    rays.push_back(Ray(p2,d2));
    rays.push_back(Ray(p3,d3));

    distances.resize(rays.size());

    #pragma acc parallel loop
    for(int i=0; i<rays.size(); i++) {
        distances[i] = g.intersect(rays[i]);
    }

    assert(distances[0] == 1.0);
    assert(distances[1] == INFINITY);
    assert(distances[2] == 0.0);

    return 0;
}

Ray::Ray(double org[3], double dir[3]) {
    for(int i=0; i<3; i++) {
        origin[i] = org[i];
        direction[i] = dir[i];
    }
}

double intersect_helper(double origin_i, double direction_i) {
    if(origin_i == 0.0) {
        return 0.0;
    } else if(direction_i == 0.0) {
        return INFINITY;
    }

    double t = -origin_i / direction_i;
    if(t <= 0.0) {
        return INFINITY;
    }
    return t;
}

double XPlane::intersect(const Ray& r) const {
    return intersect_helper(r.origin[0], r.direction[0]);
}

double YPlane::intersect(const Ray& r) const {
    return intersect_helper(r.origin[1], r.direction[1]);
}

double ZPlane::intersect(const Ray& r) const {
    return intersect_helper(r.origin[2], r.direction[2]);
}

void Geometry::add(shared_ptr<Shape> s) {
    shapes_.push_back(s);
}
    
double Geometry::intersect(const Ray& r) const {
    double min_dist = INFINITY;
    for(int i=0; i<shapes_.size(); i++) {
        min_dist = std::min(min_dist, shapes_[i]->intersect(r));
    }

    return min_dist;
}
