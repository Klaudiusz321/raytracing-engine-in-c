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
    
    // Calculate dot product manually
    double dot_product = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    printf("\nDot product: %.2f\n", dot_product);
    
    // Calculate length manually
    double length_v1 = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    printf("Length of v1: %.2f\n", length_v1);
    
    // Try to calculate normalized vector manually
    Vector3D normalized = {
        v1.x / length_v1,
        v1.y / length_v1,
        v1.z / length_v1
    };
    printf("Normalized v1: (%.2f, %.2f, %.2f)\n", 
           normalized.x, normalized.y, normalized.z);
    
    // Calculate length of the normalized vector
    double length_normalized = sqrt(
        normalized.x * normalized.x + 
        normalized.y * normalized.y + 
        normalized.z * normalized.z
    );
    printf("Length of normalized v1: %.2f\n", length_normalized);
    
    // Calculate and print some constants
    printf("\nMath Constants:\n");
    printf("PI: %.6f\n", PI);
    printf("SPEED_OF_LIGHT: %.0f\n", SPEED_OF_LIGHT);
    printf("GRAVITATIONAL_CONSTANT: %.6e\n", GRAVITATIONAL_CONSTANT);
    
    return 0;
} 