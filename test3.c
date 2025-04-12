#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "include/blackhole_types.h"
#include "include/math_util.h"

int main() {
    printf("Black Hole Physics Engine - Math Test\n");
    printf("=====================================\n\n");
    
    // Create two vectors
    Vector3D v1 = {1.0, 2.0, 3.0};
    Vector3D v2 = {4.0, 5.0, 6.0};
    
    // Test vector operations
    printf("Vector v1: (%.2f, %.2f, %.2f)\n", v1.x, v1.y, v1.z);
    printf("Vector v2: (%.2f, %.2f, %.2f)\n", v2.x, v2.y, v2.z);
    
    // Calculate dot product 
    double dot = vector3D_dot(v1, v2);
    printf("\nDot product: %.2f\n", dot);
    
    // Calculate length
    double length_v1 = vector3D_length(v1);
    printf("Length of v1: %.2f\n", length_v1);
    
    // Calculate normalized vector
    Vector3D normalized = vector3D_normalize(v1);
    printf("Normalized v1: (%.2f, %.2f, %.2f)\n", 
           normalized.x, normalized.y, normalized.z);
    
    // Calculate cross product
    Vector3D cross = vector3D_cross(v1, v2);
    printf("Cross product v1 x v2: (%.2f, %.2f, %.2f)\n",
           cross.x, cross.y, cross.z);
    
    // Calculate and print some constants
    printf("\nMath Constants:\n");
    printf("PI: %.6f\n", PI);
    printf("SPEED_OF_LIGHT: %.0f\n", SPEED_OF_LIGHT);
    printf("GRAVITATIONAL_CONSTANT: %.6e\n", GRAVITATIONAL_CONSTANT);
    
    return 0;
} 