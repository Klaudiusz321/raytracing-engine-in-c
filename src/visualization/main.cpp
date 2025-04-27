#include "./renderer.h"
#include <iostream>
#include <cstring> // For strcmp function

/**
 * Main entry point for the black hole visualization application
 */
int main(int argc, char* argv[]) {
    // Create the renderer
    Renderer renderer;
    
    // Initialize with default window size
    int width = 1280;
    int height = 720;
    const char* title = "Black Hole Physics Simulator";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--title") == 0 && i + 1 < argc) {
            title = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Black Hole Physics Simulator\n"
                      << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --width <pixels>   Set window width (default: 1280)\n"
                      << "  --height <pixels>  Set window height (default: 720)\n"
                      << "  --title <string>   Set window title\n"
                      << "  --help             Show this help message\n";
            return 0;
        }
    }
    
    // Initialize the renderer
    if (!renderer.initialize(width, height, title)) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return 1;
    }
    
    // Run the main loop
    renderer.runMainLoop();
    
    // Cleanup (will be called by renderer's destructor)
    
    return 0;
} 