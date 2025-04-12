#include <math.h>
#include "include/blackhole_types.h"
#include "include/math_util.h"

Vector3D vector3D_create(double x, double y, double z) {
    Vector3D v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

Vector4D vector4D_create(double t, double x, double y, double z) {
    Vector4D v;
    v.t = t;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

Vector3D vector3D_add(const Vector3D a, const Vector3D b) {
    Vector3D result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
}

Vector3D vector3D_sub(const Vector3D a, const Vector3D b) {
    Vector3D result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

Vector3D vector3D_scale(const Vector3D v, double scalar) {
    Vector3D result;
    result.x = v.x * scalar;
    result.y = v.y * scalar;
    result.z = v.z * scalar;
    return result;
}

double vector3D_dot(const Vector3D a, const Vector3D b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector3D vector3D_cross(const Vector3D a, const Vector3D b) {
    Vector3D result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

double vector3D_length(const Vector3D v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vector3D vector3D_normalize(const Vector3D v) {
    double length = vector3D_length(v);
    if (length < BH_EPSILON) {
        // Avoid division by zero
        return (Vector3D){0.0, 0.0, 0.0};
    }
    return vector3D_scale(v, 1.0 / length);
}

// Implement only what we need for our test
double clamp(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

void cartesian_to_spherical(const Vector3D* cartesian, Vector3D* spherical) {
    double x = cartesian->x;
    double y = cartesian->y;
    double z = cartesian->z;
    
    // Calculate r (radius)
    double r = sqrt(x*x + y*y + z*z);
    
    // Calculate theta (polar angle)
    double theta = 0.0;
    if (r > BH_EPSILON) {
        theta = acos(z / r);
    }
    
    // Calculate phi (azimuthal angle)
    double phi = atan2(y, x);
    if (phi < 0.0) {
        phi += TWO_PI;
    }
    
    spherical->x = r;
    spherical->y = theta;
    spherical->z = phi;
}

void spherical_to_cartesian(const Vector3D* spherical, Vector3D* cartesian) {
    double r = spherical->x;
    double theta = spherical->y;
    double phi = spherical->z;
    
    cartesian->x = r * sin(theta) * cos(phi);
    cartesian->y = r * sin(theta) * sin(phi);
    cartesian->z = r * cos(theta);
} 