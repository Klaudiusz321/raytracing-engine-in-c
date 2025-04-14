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

#include "../../include/blackhole_api.h"

// Forward declarations
struct Shader;
struct Camera;
struct RenderData;

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
    } m_uiState;
    
    // Particle data for rendering
    struct {
        std::vector<float> positions;
        std::vector<float> velocities;
        std::vector<int> types;
        int count;
    } m_particleData;
    
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
        position({0.0, 0.0, 30.0}),
        target({0.0, 0.0, 0.0}),
        up({0.0, 1.0, 0.0}),
        fov(45.0f),
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