/**
 * renderer.h
 * 
 * OpenGL-based renderer for the black hole simulation.
 */

#ifndef RENDERER_H
#define RENDERER_H

// Update include paths to match actual project structure
#include "../../include/gl.h"  // Using the gl.h in include directory instead of glad/glad.h
#include "../../lib/glfw-3.4.bin.WIN64/glfw-3.4.bin.WIN64/include/GLFW/glfw3.h"
#include "../../external/imgui/imgui.h"
#include "../../external/imgui/backends/imgui_impl_glfw.h"
#include "../../external/imgui/backends/imgui_impl_opengl3.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <string>
#include <fstream>
#include <sstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../../include/blackhole_api.h"

// Forward declarations
struct Shader;
struct Camera;
struct RenderData;

/**
 * Temporal accumulation buffer for reducing flicker
 * Maintains history of frames to blend over time
 */
typedef struct {
    float* accumulated_color;    // RGB accumulation buffer
    float* history_color;        // Previous frame buffer
    int width, height;           // Buffer dimensions
    int frame_count;             // Number of frames accumulated
    float blend_factor;          // How much to blend with previous frames (e.g., 0.1)
    bool reset_accumulation;     // Flag to reset on camera move or parameter change
    int max_frames;              // Maximum frames to accumulate
    int jitter_index;            // Current jitter index for progressive sampling
} TemporalAccumulationBuffer;

/**
 * GPU Compute shader resources for ray tracing
 * This handles the GPU acceleration of the ray tracing process
 */
typedef struct {
    GLuint computeProgram;        // Compute shader program
    GLuint rayInputSSBO;          // Ray input storage buffer
    GLuint rayOutputSSBO;         // Ray output storage buffer
    GLuint blackHoleParamsUBO;    // Black hole parameters uniform buffer
    int workGroupSize;            // Compute work group size
    bool initialized;             // Whether GPU compute is initialized
    int width, height;            // Current output dimensions
} GPUComputeResources;

/**
 * Structure to hold black hole parameters
 * This is the renderer's version of BlackHoleParams that adds visualization-specific fields
 */
struct RendererBlackHoleParams {
    // Black hole properties
    float mass;                 // Mass of the black hole
    float spin;                 // Spin parameter (a) - 0 for Schwarzschild, 0-1 for Kerr
    float rs;                   // Schwarzschild radius
    float r_horizon;            // Event horizon radius
    float r_isco;               // Innermost stable circular orbit
    
    // Accretion disk properties
    float disk_inner_radius;    // Inner radius of accretion disk
    float disk_outer_radius;    // Outer radius of accretion disk
    float disk_temp_scale;      // Temperature scaling factor
    float disk_density_scale;   // Density scaling factor
    float disk_inclination;     // Disk inclination angle (radians)
    
    // Observer parameters
    glm::vec3 observer_pos;     // Observer position (spherical coordinates: r, theta, phi)
    glm::vec3 observer_dir;     // Observer direction vector
    glm::vec3 up_vector;        // Camera up vector
    
    // Viewing parameters
    float fov;                  // Field of view in degrees
    float aspect_ratio;         // Aspect ratio (width/height)
    
    // Feature flags
    int enable_doppler;         // Enable Doppler effect (0/1)
    int enable_redshift;        // Enable gravitational redshift (0/1)
    int show_disk;              // Show accretion disk (0/1)
    int adaptive_stepping;      // Use adaptive stepping for integration (0/1)
    
    // Integration parameters
    int max_steps;              // Maximum number of steps for ray integration
    float step_size;            // Initial step size for integration
    float tolerance;            // Error tolerance for adaptive stepping
    float max_distance;         // Maximum ray travel distance
    float celestial_sphere_radius; // Radius of the background celestial sphere
};

/**
 * Main renderer class that handles the window, OpenGL context,
 * and integration with the physics engine.
 */
class Renderer {
public:
    Renderer();
    ~Renderer();

    /**
     * Initialize the renderer and create a window
     * 
     * @param width Initial window width
     * @param height Initial window height
     * @param title Window title
     * @return true if initialization succeeded, false otherwise
     */
    bool initialize(int width, int height, const char* title);

    /**
     * Main rendering loop
     */
    void runMainLoop();

    /**
     * Shut down the renderer and release resources
     */
    void shutdown();

    // Method to update ray tracing texture
    void updateRayTraceTexture();

    // Method to create a visually enhanced black hole overlay
    void drawBlackHoleOverlay();

    // Initialization
    bool init(int windowWidth, int windowHeight);
    
    // Main rendering function
    void render();
    
    // Window resize handler
    void resize(int newWidth, int newHeight);
    
    // Parameter setters
    void setBlackHoleParams(const RendererBlackHoleParams& params);
    
    // Getters
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    RendererBlackHoleParams getBlackHoleParams() const { return blackHoleParams; }

private:
    // Window and context
    GLFWwindow* m_window;
    int m_width;
    int m_height;
    
    // Physics engine
    BHContextHandle m_physicsContext;
    std::thread m_physicsThread;
    std::atomic<bool> m_running;
    
    // Render data with double-buffering
    std::mutex m_renderDataMutex;
    RenderData* m_renderDataFront; // For rendering
    RenderData* m_renderDataBack;  // For physics updates
    std::condition_variable m_dataReadyCV;
    bool m_dataReady;

    // Shaders and render objects
    Shader* m_blackHoleShader;
    Shader* m_accretionDiskShader;
    Shader* m_skyboxShader;
    
    // Camera
    Camera* m_camera;
    Vector3D m_previousCameraPosition; // For detecting camera movement
    
    // UI state
    struct {
        float mass; // Solar masses
        float spin; // a/M
        float accretionRate; // M-dot
        bool showAccretionDisk;
        bool showGrid;
        bool showStars;
        bool enableDopplerEffect;
        bool enableGravitationalRedshift;
        int integrationSteps; // For ray tracing
        float timeScale; // Simulation speed
        
        // New antialiasing and quality settings
        int samplesPerPixel; // 1, 4, 16
        int jitterMethod;    // 0=None, 1=Regular, 2=Random, 3=Halton
        bool enableAdaptiveSampling;
        bool enableTemporalAccumulation;
        float convergenceThreshold;
        
        // GPU Acceleration
        bool useGPUAcceleration;
        int workGroupSize;   // GPU work group size (e.g., 8x8, 16x16)
    } m_uiState;
    
    // Particle data for rendering
    struct {
        std::vector<float> positions;
        std::vector<float> velocities;
        std::vector<int> types;
        int count;
    } m_particleData;
    
    // Ray tracing resources
    static GLuint m_rayTraceTextureID;
    static unsigned char* m_rayTraceData;
    static int m_rtWidth;
    static int m_rtHeight;
    GLuint m_quadVAO; // VAO for fullscreen quad
    
    // New: Temporal accumulation buffer for flicker reduction
    TemporalAccumulationBuffer m_temporalBuffer;
    float* m_edgeFactorBuffer; // For adaptive sampling
    
    // New: GPU compute resources for ray tracing
    GPUComputeResources m_computeResources;
    
    // Renderer dimensions
    int width;
    int height;
    
    // Render target
    GLuint renderTexture;
    
    // Black hole parameters
    RendererBlackHoleParams blackHoleParams;
    
    /**
     * Initialize OpenGL and create context
     */
    bool initOpenGL();
    
    /**
     * Initialize ImGui
     */
    void initImGui();
    
    /**
     * Initialize shaders
     */
    bool initShaders();
    
    /**
     * Initialize particle shader
     */
    bool initParticleShader();
    
    /**
     * Initialize the physics engine
     */
    bool initPhysics();
    
    /**
     * Render a single frame
     */
    void renderFrame();
    
    /**
     * Render the UI
     */
    void renderUI();
    
    /**
     * Render accretion disk particles
     */
    void renderAccretionDiskParticles();
    
    /**
     * Update camera based on input
     */
    void updateCamera(float deltaTime);
    
    /**
     * Physics thread function
     */
    void physicsThreadFunc();
    
    /**
     * Swap render data buffers
     */
    void swapRenderBuffers();
    
    /**
     * Update physics parameters from UI
     */
    void updatePhysicsParams();
    
    /**
     * Window resize callback
     */
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    
    // Method to create and setup the ray tracing texture
    bool initRayTraceTexture();
    
    // Method to clean up ray tracing resources
    void cleanupRayTrace();
    
    // Method to create full screen quad for rendering
    void createFullScreenQuad();
    
    // Method to clean up buffer objects
    void cleanupBuffers();
    
    // Method to clean up shader objects
    void cleanupShaders();

    // New: Initialize temporal accumulation buffer
    bool initTemporalBuffer(int width, int height);
    
    // New: Accumulate current frame into temporal buffer
    void accumulateFrame(const float* newFrame);
    
    // New: Reset temporal accumulation (on camera/parameter change)
    void resetAccumulation();
    
    // New: Detect high gradient regions for adaptive sampling
    void detectEdges();
    
    // New: Convert accumulated buffer to output texture
    void finalizeAccumulatedFrame();
    
    // New: Initialize GPU compute resources
    bool initComputeResources();
    
    // New: Update GPU compute resources (when parameters change)
    void updateComputeResources();
    
    // New: Execute the ray tracing on the GPU
    void executeRayTracingOnGPU();
    
    // New: Clean up GPU compute resources
    void cleanupComputeResources();

    // Load background texture
    bool loadBackgroundTexture(const std::string& filename);
    
    // Initialize framebuffer
    void initFramebuffer();

    // Draw fullscreen quad with result
    void drawFullscreenQuad();

    // Helper function to read shader file
    std::string readShaderFile(const std::string& filePath);
};

/**
 * Render data structure (double-buffered)
 */
struct RenderData {
    FrameData frameData;
    std::vector<Vector3D> particlePositions;
    std::vector<Vector3D> particleVelocities;
    std::vector<int> particleTypes;
    double simulationTime;
    
    RenderData() : simulationTime(0.0) {}
    
    void clear() {
        particlePositions.clear();
        particleVelocities.clear();
        particleTypes.clear();
        simulationTime = 0.0;
    }
};

/**
 * Camera class
 */
struct Camera {
    Vector3D position;
    Vector3D target;
    Vector3D up;
    float fov;
    float nearPlane;
    float farPlane;
    
    Camera() : 
        position({0.0, 0.0, 75.0}),  // Move from 50.0 to 75.0 - much further from the black hole
        target({0.0, 0.0, 0.0}),
        up({0.0, 1.0, 0.0}),
        fov(40.0f),                  // Narrower field of view (less rays)
        nearPlane(0.1f),
        farPlane(10000.0f) {}
};

/**
 * Shader class
 */
struct Shader {
    GLuint program;
    
    void use() {
        glUseProgram(program);
    }
    
    void setInt(const char* name, int value) {
        glUniform1i(glGetUniformLocation(program, name), value);
    }
    
    void setFloat(const char* name, float value) {
        glUniform1f(glGetUniformLocation(program, name), value);
    }
    
    void setVec3(const char* name, float x, float y, float z) {
        glUniform3f(glGetUniformLocation(program, name), x, y, z);
    }
    
    void setVec3(const char* name, const Vector3D& vec) {
        glUniform3f(glGetUniformLocation(program, name), 
                    (float)vec.x, (float)vec.y, (float)vec.z);
    }
    
    void setMat4(const char* name, const float* mat) {
        glUniformMatrix4fv(glGetUniformLocation(program, name), 1, GL_FALSE, mat);
    }
};

#endif /* RENDERER_H */ 