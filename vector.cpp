#include<iostream>

class vector_ {
private:
    double x, y, z;

public:
    vector_(): vector_(0, 0, 0) { }
    vector_(double x_, double y_, double z_)
    :x(x_), y(y_), z(z_) {}

    double x() const { return x; }
    double y() const { return y; }
    double z() const { return z; }

    void x (double x) { this->x = x; }
    void y (double y) { this->y = y; }
    void z (double z) { this->z = z; }
};