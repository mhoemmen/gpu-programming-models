#include<algorithm>
#include<cassert>
#include<execution>
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
    Shape(double(*f)(const Ray&)) : f_(f) {}

    // Returns the distance to the intersection point
    // Infinity if there is no intersection
    double intersect(const Ray& r) const { return f_(r); }

private:
    double(*f_)(const Ray&);
};

double x_intersect(const Ray& r);
double y_intersect(const Ray& r);
double z_intersect(const Ray& r);

// The plane x=0
class XPlane : public Shape {
public:
    XPlane() : Shape(x_intersect) {}
};

// The plane y=0
class YPlane : public Shape {
public:
    YPlane() : Shape(y_intersect) {}
};

// The plane z=0
class ZPlane : public Shape {
public:
    ZPlane() : Shape(z_intersect) {}
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
    using std::execution::par_unseq;

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

    std::transform(par_unseq, rays.cbegin(), rays.cend(), distances.begin(), [=](const Ray& ray) {
        return g.intersect(ray);
    });

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

double x_intersect(const Ray& r) {
    return intersect_helper(r.origin[0], r.direction[0]);
}

double y_intersect(const Ray& r) {
    return intersect_helper(r.origin[1], r.direction[1]);
}

double z_intersect(const Ray& r) {
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
