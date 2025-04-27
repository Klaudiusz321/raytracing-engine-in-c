#include "../../include/gl.h"
#include <GLFW/glfw3.h>
#include "../../external/glm/glm/glm.hpp"
#include "../../external/glm/glm/gtc/matrix_transform.hpp"
#include "../../external/glm/glm/gtc/type_ptr.hpp"
#include "renderer.h"
#include "../../include/blackhole_api.h"
#include "../../include/raytracer.h"
#include "../../include/blackhole_types.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>

// Define constants that are missing
#define GL_COMPUTE_SHADER               0x91B9
#define GL_SHADER_STORAGE_BUFFER        0x90D2
#define GL_SHADER_STORAGE_BARRIER_BIT   0x00002000
#define GL_MAX_COMPUTE_WORK_GROUP_COUNT     0x91BE
#define GL_MAX_COMPUTE_WORK_GROUP_SIZE      0x91BF
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB

// Function pointers for OpenGL 4.3 Compute Shader functions
typedef void (GLAD_API_PTR *PFNGLDISPATCHCOMPUTEPROC)(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
typedef void (GLAD_API_PTR *PFNGLMEMORYBARRIERPROC)(GLbitfield barriers);

// Function pointers
PFNGLDISPATCHCOMPUTEPROC glDispatchCompute;
PFNGLMEMORYBARRIERPROC glMemoryBarrier;

// Function to dynamically load the compute shader functions
bool loadComputeShaderFunctions() {
    // Load compute shader functions using OpenGL's function pointers
    glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)glfwGetProcAddress("glDispatchCompute");
    glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)glfwGetProcAddress("glMemoryBarrier");
    
    // Check if the pointers were successfully loaded
    if (!glDispatchCompute || !glMemoryBarrier) {
        printf("Failed to load compute shader functions\n");
        return false;
    }
    
    printf("Successfully loaded compute shader functions\n");
    return true;
}

// For Windows-specific sleep function
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Initialize static members
GLuint Renderer::m_rayTraceTextureID = 0;
unsigned char* Renderer::m_rayTraceData = nullptr;
int Renderer::m_rtWidth = 512;
int Renderer::m_rtHeight = 512;

// Add this member variable and static shader ID
static GLuint m_particleShaderProgram = 0;

// Shader source code for the black hole ray tracing
const char* blackHoleVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)";

// Fragment shader that uses precomputed ray traced data
const char* blackHoleFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

// Ray traced texture
uniform sampler2D rayTraceTexture;
uniform float time;
uniform vec2 resolution;

// Black hole parameters
uniform float blackHoleMass;
uniform float blackHoleRadius;
uniform float diskInnerRadius;
uniform float diskOuterRadius;
uniform float accretionRate;
uniform bool enableDoppler;
uniform bool enableRedshift;

// Gravitational lensing function
vec2 gravitationalLensing(vec2 texCoord) {
    // Convert texture coordinates to normalized device coordinates (-1 to 1)
    vec2 ndc = texCoord * 2.0 - 1.0;
    
    // Calculate distance from center of screen
    float r = length(ndc);
    
    // Simple gravitational lensing effect based on distance
    // This is a simplified model for visual enhancement
    float lensStrength = 0.2 * blackHoleMass;
    float lensEffect = 1.0 / (1.0 + lensStrength * pow(max(0.0, 1.0 - r), 3.0));
    
    // Apply distortion
    vec2 lensedCoord = texCoord + (texCoord - 0.5) * (lensEffect - 1.0);
    
    return lensedCoord;
}

// Disk temperature function (simplified accretion physics)
vec3 getDiskColor(float radius) {
    // Temperature follows T ~ r^(-3/4) for thin accretion disk
    float temperature = 1.0 - pow(diskInnerRadius / max(radius, diskInnerRadius + 0.1), 0.75);
    
    // Map temperature to color (red->yellow->white)
    vec3 color;
    if (temperature < 0.5) {
        // Red to yellow
        color = vec3(1.0, temperature * 2.0, 0.0);
    } else {
        // Yellow to white
        color = vec3(1.0, 1.0, (temperature - 0.5) * 2.0);
    }
    
    return color * accretionRate;
}

// Doppler shift effect 
vec3 applyDopplerEffect(vec3 color, vec2 diskCoord) {
    if (!enableDoppler) 
        return color;
        
    // Angle from center determines if material is moving toward or away
    float angle = atan(diskCoord.y, diskCoord.x);
    
    // Calculate Doppler shift factor (simplified model)
    // Left side moving toward, right side moving away
    float dopplerFactor = 1.0 + 0.3 * cos(angle);
    
    // Apply Doppler shift to colors (blue/red shift)
    return vec3(
        color.r * (dopplerFactor > 1.0 ? 1.0 : 1.0/dopplerFactor),
        color.g,
        color.b * (dopplerFactor < 1.0 ? 1.0 : dopplerFactor)
    );
}

// Main function
void main() {
    // Sample from precomputed ray trace texture
    vec4 rayTraceColor = texture(rayTraceTexture, TexCoord);
    
    // Alpha channel encodes hit type:
    // 0.0 = black hole
    // 1.0 = accretion disk
    // 0.5 = background/stars
    
    float hitType = rayTraceColor.a;
    
    if (hitType < 0.01) {
        // Black hole - pure black with subtle edge glow
        float distToCenter = length(TexCoord - 0.5) * 2.0;
        float edgeGlow = smoothstep(0.2, 0.3, distToCenter) * 0.15;
        vec3 glowColor = vec3(0.7, 0.3, 0.1) * edgeGlow;
        FragColor = vec4(glowColor, 1.0);
    }
    else if (abs(hitType - 1.0) < 0.01) {
        // Accretion disk - use precomputed color with additional visual effects
        vec3 diskColor = rayTraceColor.rgb;
        
        // Add time-based variation for more dynamic appearance
        float brightnessPulse = 1.0 + 0.1 * sin(time * 0.5);
        diskColor *= brightnessPulse;
        
        // Add subtle shimmer effect
        float shimmer = 1.0 + 0.05 * sin(time * 3.0 + TexCoord.x * 20.0 + TexCoord.y * 15.0);
        diskColor *= shimmer;
        
        FragColor = vec4(diskColor, 1.0);
    }
    else {
        // Background - use precomputed color with added visual enhancements
        vec3 bgColor = rayTraceColor.rgb;
        
        // Apply subtle distortion to stars based on proximity to black hole
        float distToCenter = length(TexCoord - 0.5) * 2.0;
        float distortionFactor = 0.15 * blackHoleMass / max(0.2, distToCenter);
        vec2 distortedCoord = TexCoord + (TexCoord - 0.5) * distortionFactor;
        
        // Only apply the distortion to actual stars (bright pixels)
        float luminance = dot(bgColor, vec3(0.2126, 0.7152, 0.0722));
        if (luminance > 0.3) { // Only apply to stars
            vec3 distortedColor = texture(rayTraceTexture, distortedCoord).rgb;
            float blendFactor = smoothstep(0.3, 0.6, distToCenter);
            bgColor = mix(distortedColor, bgColor, blendFactor);
        }
        
        FragColor = vec4(bgColor, 1.0);
    }
}
)";

Renderer::Renderer() :
    m_window(nullptr),
    m_width(0),
    m_height(0),
    m_physicsContext(nullptr),
    m_running(false),
    m_renderDataFront(nullptr),
    m_renderDataBack(nullptr),
    m_dataReady(false),
    m_blackHoleShader(nullptr),
    m_accretionDiskShader(nullptr),
    m_skyboxShader(nullptr),
    m_camera(nullptr),
    m_quadVAO(0),
    m_edgeFactorBuffer(nullptr)
{
    // Initialize UI state
    m_uiState.mass = 1.0f;
    m_uiState.spin = 0.0f;
    m_uiState.accretionRate = 1.0f;
    m_uiState.showAccretionDisk = true;
    m_uiState.showGrid = false;
    m_uiState.showStars = true;
    m_uiState.enableDopplerEffect = true;
    m_uiState.enableGravitationalRedshift = true;
    m_uiState.integrationSteps = 1000;
    m_uiState.timeScale = 1.0f;
    
    // Initialize antialiasing and quality settings
    m_uiState.samplesPerPixel = 4;
    m_uiState.jitterMethod = 3; // Halton sequence
    m_uiState.enableAdaptiveSampling = true;
    m_uiState.enableTemporalAccumulation = true;
    m_uiState.convergenceThreshold = 0.01f;
    
    // Initialize temporal buffer struct
    m_temporalBuffer.accumulated_color = nullptr;
    m_temporalBuffer.history_color = nullptr;
    m_temporalBuffer.width = 0;
    m_temporalBuffer.height = 0;
    m_temporalBuffer.frame_count = 0;
    m_temporalBuffer.blend_factor = 0.1f;
    m_temporalBuffer.reset_accumulation = true;
    m_temporalBuffer.max_frames = 32;
    m_temporalBuffer.jitter_index = 0;
}

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::initialize(int width, int height, const char* title) {
        m_width = width;
        m_height = height;
        
        // Initialize OpenGL
        if (!initOpenGL()) {
            return false;
        }
        
        // Initialize ImGui
        initImGui();
        
        // Initialize shaders
        if (!initShaders()) {
        return false;
    }
    
    // Initialize physics engine
    if (!initPhysics()) {
            return false;
        }
        
        // Initialize ray trace texture
        if (!initRayTraceTexture()) {
            return false;
        }
        
    // Initialize temporal accumulation buffer
    if (!initTemporalBuffer(width, height)) {
        return false;
    }
    
    // Create full screen quad
        createFullScreenQuad();
        
    // Create render data buffers
    m_renderDataFront = new RenderData();
    m_renderDataBack = new RenderData();
    
    // Initialize camera
    m_camera = new Camera();
    
    // Start physics thread
    m_running = true;
    m_physicsThread = std::thread(&Renderer::physicsThreadFunc, this);
    
        return true;
}

void Renderer::runMainLoop() {
    // Main loop
    while (!glfwWindowShouldClose(m_window)) {
        // Calculate delta time
        static auto lastFrameTime = std::chrono::high_resolution_clock::now();
        auto currentFrameTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentFrameTime - lastFrameTime).count();
        lastFrameTime = currentFrameTime;
            
            // Poll events
            glfwPollEvents();
            
        // Update camera
        updateCamera(deltaTime);
            
            // Render frame
            renderFrame();
            
        // Update temporal accumulation buffer with new frame
        // In a complete implementation, this would use the new ray-traced frame
            
            // Swap buffers
            glfwSwapBuffers(m_window);
    }
}

void Renderer::shutdown() {
    // Stop physics thread
    m_running = false;
    if (m_physicsThread.joinable()) {
        m_physicsThread.join();
    }
    
    // Clean up ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // Clean up render data
    if (m_renderDataFront) {
    delete m_renderDataFront;
    m_renderDataFront = nullptr;
    }
    if (m_renderDataBack) {
        delete m_renderDataBack;
    m_renderDataBack = nullptr;
    }
    
    // Clean up camera
    if (m_camera) {
    delete m_camera;
    m_camera = nullptr;
    }
    
    // Clean up ray trace data
    cleanupRayTrace();
    
    // Clean up temporal buffer
    if (m_temporalBuffer.accumulated_color) {
        delete[] m_temporalBuffer.accumulated_color;
        m_temporalBuffer.accumulated_color = nullptr;
    }
    if (m_temporalBuffer.history_color) {
        delete[] m_temporalBuffer.history_color;
        m_temporalBuffer.history_color = nullptr;
    }
    if (m_edgeFactorBuffer) {
        delete[] m_edgeFactorBuffer;
        m_edgeFactorBuffer = nullptr;
    }
    
    // Clean up shaders
    cleanupShaders();
    
    // Clean up GLFW
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

bool Renderer::initOpenGL() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // Set OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create window
    m_window = glfwCreateWindow(m_width, m_height, "Black Hole Visualizer", NULL, NULL);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    // Make the window's context current
    glfwMakeContextCurrent(m_window);
    
    // Set up resize callback
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);

    // Load OpenGL functions with GLAD
    // Note: We already initialized GLFW and created the window in initialize()
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }
    
    // Configure OpenGL
    glViewport(0, 0, m_width, m_height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Print OpenGL info
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    std::cout << "Renderer: " << (renderer ? (const char*)renderer : "Unknown") << std::endl;
    std::cout << "OpenGL version: " << (version ? (const char*)version : "Unknown") << std::endl;
    
    return true;
}

void Renderer::initImGui() {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    
    // Enable keyboard controls and gamepad controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Make UI more visible with larger fonts and controls
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(10, 8);
    style.ScrollbarSize = 20.0f;
    style.ScrollbarRounding = 5.0f;
    
    // Increase font size for better readability
    ImFontConfig fontConfig;
    fontConfig.SizePixels = 16.0f;
    io.Fonts->AddFontDefault(&fontConfig);
    
    // Setup colors for dark theme with better contrast
    ImVec4* colors = ImGui::GetStyle().Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.10f, 0.95f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.20f, 1.00f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.15f, 0.15f, 0.30f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.40f, 0.60f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.30f, 0.30f, 0.70f, 1.00f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.40f, 0.40f, 0.80f, 1.00f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.40f, 0.40f, 0.80f, 0.30f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.40f, 0.40f, 0.80f, 0.60f);
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    printf("ImGui initialized successfully\n");
}

bool Renderer::initShaders() {
    // Allocate shader objects
    m_blackHoleShader = new Shader();
    
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &blackHoleVertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check for errors
    GLint success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Black hole vertex shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &blackHoleFragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    // Check for errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Black hole fragment shader compilation failed: " << infoLog << std::endl;
        glDeleteShader(vertexShader);
        return false;
    }
    
    // Link shaders
    m_blackHoleShader->program = glCreateProgram();
    glAttachShader(m_blackHoleShader->program, vertexShader);
    glAttachShader(m_blackHoleShader->program, fragmentShader);
    glLinkProgram(m_blackHoleShader->program);
    
    // Check for errors
    glGetProgramiv(m_blackHoleShader->program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_blackHoleShader->program, 512, NULL, infoLog);
        std::cerr << "Black hole shader program linking failed: " << infoLog << std::endl;
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return false;
    }
    
    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    // Also initialize particle shader
    if (!initParticleShader()) {
        std::cerr << "Failed to initialize particle shader" << std::endl;
        return false;
    }
    
    return true;
}

bool Renderer::initParticleShader() {
    // Create simple point shader for particles
    static const char* particleVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 projection;
    uniform mat4 view;
    
    void main() {
        gl_Position = projection * view * vec4(aPos, 1.0);
        gl_PointSize = 6.0; // Make particles even bigger for visibility
    }
    )";
    
    static const char* particleFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    uniform vec3 particleColor;
    
    void main() {
        // Create a circular point with soft edges
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord) * 2.0;
        float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
        
        // Apply color with alpha for glow effect
        FragColor = vec4(particleColor, alpha);
    }
    )";
    
    // Compile particle vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &particleVertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check for errors
    GLint success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Particle vertex shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    // Compile particle fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &particleFragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    // Check for errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Particle fragment shader compilation failed: " << infoLog << std::endl;
        glDeleteShader(vertexShader);
        return false;
    }
    
    // Link shaders
    m_particleShaderProgram = glCreateProgram();
    glAttachShader(m_particleShaderProgram, vertexShader);
    glAttachShader(m_particleShaderProgram, fragmentShader);
    glLinkProgram(m_particleShaderProgram);
    
    // Check for errors
    glGetProgramiv(m_particleShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_particleShaderProgram, 512, NULL, infoLog);
        std::cerr << "Particle shader program linking failed: " << infoLog << std::endl;
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return false;
    }
    
    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return true;
}

bool Renderer::initPhysics() {
    // Initialize the black hole physics engine
    m_physicsContext = bh_initialize();
    
    if (!m_physicsContext) {
        std::cerr << "Failed to initialize physics engine" << std::endl;
        return false;
    }
    
    // Configure the black hole
    BHErrorCode error = bh_configure_black_hole(
        m_physicsContext,
        m_uiState.mass,
        m_uiState.spin,
        0.0  // No charge for now
    );
    
    if (error != BH_SUCCESS) {
        std::cerr << "Failed to configure black hole parameters" << std::endl;
        return false;
    }
    
    // Configure the accretion disk
    error = bh_configure_accretion_disk(
        m_physicsContext,
        6.0,  // Inner radius at ISCO
        20.0, // Outer radius
        1.0,  // Temperature scale
        m_uiState.accretionRate // Density scale
    );
    
    if (error != BH_SUCCESS) {
        std::cerr << "Failed to configure accretion disk" << std::endl;
        return false;
    }
    
    // Configure simulation parameters
    error = bh_configure_simulation(
        m_physicsContext,
        0.01,  // Time step
        100.0, // Max ray distance
        m_uiState.integrationSteps, // Integration steps
        1e-6   // Tolerance
    );
    
    if (error != BH_SUCCESS) {
        std::cerr << "Failed to configure simulation parameters" << std::endl;
        return false;
    }
    
    return true;
}

void Renderer::renderFrame() {
    // Clear the screen
    glClearColor(0.0f, 0.0f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Update the ray trace texture (but not every frame)
    static int frameCount = 0;
    frameCount++;
    if (frameCount % 3 == 0) { // Only update every 3rd frame for better UI responsiveness
        updateRayTraceTexture();
    }
    
    // Render the ray trace texture to the full screen quad
    if (m_rayTraceTextureID) {
        // Disable depth test for fullscreen quad
        glDisable(GL_DEPTH_TEST);
        
        // Use a very simple shader for the quad
        glUseProgram(m_blackHoleShader->program);
        
        // Set texture uniform
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_rayTraceTextureID);
        glUniform1i(glGetUniformLocation(m_blackHoleShader->program, "rayTraceTexture"), 0);
        
        // Render the quad
        glBindVertexArray(m_quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        
        // Re-enable depth test for 3D rendering
        glEnable(GL_DEPTH_TEST);
    }
    
    // Start rendering ImGui
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Render UI
    renderUI();
    
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Renderer::renderUI() {
    // Create a window for black hole parameters
    ImGui::Begin("Black Hole Parameters", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    // Black hole mass slider (solar masses)
    ImGui::Text("Mass (M)");
    ImGui::SliderFloat("##Mass", &m_uiState.mass, 0.1f, 2.0f, "%.1f");
    
    // Black hole spin slider (dimensionless a/M)
    ImGui::Text("Spin (a/M)");
    ImGui::SliderFloat("##Spin", &m_uiState.spin, 0.0f, 0.999f, "%.3f");
    
    // Accretion rate slider
    ImGui::Text("Accretion Rate");
    ImGui::SliderFloat("##AccretionRate", &m_uiState.accretionRate, 0.1f, 2.0f, "%.2f");
    
    // Checkboxes for display options
    ImGui::Checkbox("Show Accretion Disk", &m_uiState.showAccretionDisk);
    ImGui::Checkbox("Show Grid", &m_uiState.showGrid);
    ImGui::Checkbox("Show Stars", &m_uiState.showStars);
    
    // Checkboxes for relativistic effects
    ImGui::Checkbox("Doppler Effect", &m_uiState.enableDopplerEffect);
    ImGui::Checkbox("Gravitational Redshift", &m_uiState.enableGravitationalRedshift);
    
    // Slider for integration steps
    ImGui::Text("Integration Steps");
    ImGui::SliderInt("##IntegrationSteps", &m_uiState.integrationSteps, 10, 1000);
    
    // Slider for time scale
    ImGui::Text("Time Scale");
    ImGui::SliderFloat("##TimeScale", &m_uiState.timeScale, 0.1f, 10.0f, "%.1fx");
    
    // Display some stats about the black hole
    ImGui::Separator();
    ImGui::Text("Shadow Radius: %.2f M", 2.6f * m_uiState.mass); // Approximate for non-rotating BH
    ImGui::Text("ISCO Radius: %.2f M", 6.0f * m_uiState.mass); // For non-rotating BH
    
    // Display camera position
    ImGui::Text("Observer Position: (%.1f, %.1f, %.1f) M", 
                m_camera->position.x, m_camera->position.y, m_camera->position.z);
    
    // Display simulation time
    static float simTime = 0.0f;
    simTime += ImGui::GetIO().DeltaTime * m_uiState.timeScale;
    ImGui::Text("Simulation Time: %.2f M", simTime);
    
    // Display particle count with bounds checking to prevent negative display
    ImGui::Text("Particle Count: %d", std::max(0, m_particleData.count));
    
    ImGui::End();
    
    // Create a window for camera controls
    ImGui::Begin("Camera Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    // Camera position controls
    static float cameraDistance = 75.0f;
    if (ImGui::SliderFloat("Distance", &cameraDistance, 10.0f, 150.0f)) {
        // Update camera position based on distance
        Vector3D direction = vector3D_normalize(vector3D_sub(m_camera->position, m_camera->target));
        m_camera->position = vector3D_add(m_camera->target, 
                                         vector3D_scale(direction, cameraDistance));
    }
    
    // Camera FOV control
    static float cameraFOV = m_camera->fov;
    if (ImGui::SliderFloat("Field of View", &cameraFOV, 20.0f, 90.0f)) {
        m_camera->fov = cameraFOV;
    }
    
    // Display FPS
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    
    // Display instructions
    ImGui::Separator();
    ImGui::Text("Controls:");
    ImGui::BulletText("WASD: Move camera");
    ImGui::BulletText("Mouse: Look around");
    ImGui::BulletText("Escape: Exit");
    
    ImGui::End();
    
    // Create a visual debug/help window
    ImGui::Begin("Render Quality", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    // Display ray trace quality
    static int currentQuality = 1;
    ImGui::Text("Current render quality: 1/%d", currentQuality);
    ImGui::ProgressBar(1.0f / currentQuality, ImVec2(-1, 0), "Quality");
    
    // Add a button to force full quality render
    if (ImGui::Button("Force High Quality Render")) {
        // This will be handled in updateRayTraceTexture
        currentQuality = 1;
    }
    
    ImGui::End();
}

void Renderer::updateCamera(float deltaTime) {
    if (!m_camera) {
        return;
    }
    
    // Check if camera has moved significantly
    Vector3D position_diff;
    position_diff.x = m_camera->position.x - m_previousCameraPosition.x;
    position_diff.y = m_camera->position.y - m_previousCameraPosition.y;
    position_diff.z = m_camera->position.z - m_previousCameraPosition.z;
    
    const float move_threshold = 0.1f;
    float distance_sq = position_diff.x * position_diff.x + 
                        position_diff.y * position_diff.y + 
                        position_diff.z * position_diff.z;
    
    if (distance_sq > move_threshold * move_threshold) {
        // Camera has moved - reset accumulation
        m_temporalBuffer.reset_accumulation = true;
    }
    
    // Update previous camera position
    m_previousCameraPosition = m_camera->position;
}

void Renderer::physicsThreadFunc() {
    // Configure simulation with enhanced parameters
    BHErrorCode error = bh_configure_simulation(
        m_physicsContext,
        0.01,                          // Time step
        200.0,                         // Max ray distance - increased
        std::max(300, m_uiState.integrationSteps), // More integration steps for accuracy
        1e-7                           // Tighter tolerance
    );
    
    if (error != BH_SUCCESS) {
        std::cerr << "Failed to configure simulation" << std::endl;
        return;
    }
    
    // Create a particle system
    void* particleSystem = bh_create_particle_system(m_physicsContext, 5000);  // Increased capacity
    
    if (!particleSystem) {
        std::cerr << "Failed to create particle system" << std::endl;
        return;
    }
    
    // Configure disk with more physically accurate parameters
    BHErrorCode diskError = bh_configure_accretion_disk(
        m_physicsContext,
        3.0 * m_uiState.mass,          // Inner radius closer to ISCO
        30.0,                          // Outer radius
        2.0,                           // Higher temperature scale for brighter disk
        m_uiState.accretionRate        // Density scale
    );
    
    if (diskError != BH_SUCCESS) {
        std::cerr << "Failed to configure accretion disk" << std::endl;
        bh_destroy_particle_system(m_physicsContext, particleSystem);
        return;
    }
    
    // Create accretion disk particles
    int numParticles = bh_create_accretion_disk_particles(m_physicsContext, particleSystem, 3000);
    
    if (numParticles <= 0) {
        std::cerr << "Failed to create accretion disk particles" << std::endl;
        bh_destroy_particle_system(m_physicsContext, particleSystem);
        return;
    }
    
    std::cout << "Created " << numParticles << " accretion disk particles" << std::endl;
    
    // Buffers for particle data
    std::vector<double> positions(numParticles * 3);
    std::vector<double> velocities(numParticles * 3);
    std::vector<int> types(numParticles);
    
    // Physics loop
    double simulationTime = 0.0;
    double timeStep = 0.01; // Base time step
    
    while (m_running) {
        // Apply time scale from UI
        double scaledTimeStep = timeStep * m_uiState.timeScale;
        
        // Update particles using existing API
        error = bh_update_particles(m_physicsContext, particleSystem);
        
        if (error != BH_SUCCESS) {
            std::cerr << "Error updating particles" << std::endl;
            break;
        }
        
        // Update simulation time
        simulationTime += scaledTimeStep;
        
        // Get particle data for rendering
        int count = numParticles;

        // Add safety bounds for memory allocation
        std::vector<double> positions(count * 3, 0.0);
        std::vector<double> velocities(count * 3, 0.0);
        std::vector<int> types(count, 0);

        error = bh_get_particle_data(
            m_physicsContext,
            particleSystem,
            positions.data(),
            velocities.data(),
            types.data(),
            &count
        );

        if (error != BH_SUCCESS) {
            std::cerr << "Error getting particle data" << std::endl;
            break;
        }

        // Validate count to prevent overflow or negative values
        count = std::max(0, std::min(count, numParticles));
        
        // Update the back buffer with new data
        {
            std::lock_guard<std::mutex> lock(m_renderDataMutex);
            
            // Clear back buffer
            m_renderDataBack->clear();
            
            // Update simulation time
            m_renderDataBack->simulationTime = simulationTime;
            
            // Copy particle data to back buffer
            for (int i = 0; i < count; ++i) {
                Vector3D position = {
                    positions[i * 3 + 0],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2]
                };
                
                Vector3D velocity = {
                    velocities[i * 3 + 0],
                    velocities[i * 3 + 1],
                    velocities[i * 3 + 2]
                };
                
                m_renderDataBack->particlePositions.push_back(position);
                m_renderDataBack->particleVelocities.push_back(velocity);
                m_renderDataBack->particleTypes.push_back(types[i]);
            }
            
            std::cout << "Updated particle data: " << m_renderDataBack->particlePositions.size() << " particles" << std::endl;
            
            // Swap buffers
            swapRenderBuffers();
            
            // Signal that data is ready
            m_dataReady = true;
            m_dataReadyCV.notify_one();
        }
        
        // Sleep to avoid maxing out CPU - slow down updates for better debugging
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Cleanup
    bh_destroy_particle_system(m_physicsContext, particleSystem);
}

void Renderer::swapRenderBuffers() {
    // Swap front and back buffers
    RenderData* temp = m_renderDataFront;
    m_renderDataFront = m_renderDataBack;
    m_renderDataBack = temp;
}

void Renderer::updatePhysicsParams() {
    // Update black hole parameters if UI changed
    static float lastMass = m_uiState.mass;
    static float lastSpin = m_uiState.spin;
    static float lastAccretionRate = m_uiState.accretionRate;
    static bool lastDopplerEffect = m_uiState.enableDopplerEffect;
    static bool lastRedshift = m_uiState.enableGravitationalRedshift;
    static bool lastShowDisk = m_uiState.showAccretionDisk;
    
    bool parametersChanged = false;
    
    if (lastMass != m_uiState.mass || 
        lastSpin != m_uiState.spin) {
        
        // Update black hole parameters
        BHErrorCode error = bh_configure_black_hole(
            m_physicsContext,
            m_uiState.mass,
            m_uiState.spin,
            0.0  // No charge
        );
        
        if (error != BH_SUCCESS) {
            std::cerr << "Failed to update black hole parameters" << std::endl;
        }
        
        lastMass = m_uiState.mass;
        lastSpin = m_uiState.spin;
        parametersChanged = true;
    }
    
    if (lastAccretionRate != m_uiState.accretionRate) {
        // Update accretion disk parameters
        BHErrorCode error = bh_configure_accretion_disk(
            m_physicsContext,
            6.0,  // Inner radius at ISCO
            20.0, // Outer radius
            1.0,  // Temperature scale
            m_uiState.accretionRate // Density scale
        );
        
        if (error != BH_SUCCESS) {
            std::cerr << "Failed to update accretion disk parameters" << std::endl;
        }
        
        lastAccretionRate = m_uiState.accretionRate;
        parametersChanged = true;
    }
    
    // Check if visualization options changed
    if (lastDopplerEffect != m_uiState.enableDopplerEffect ||
        lastRedshift != m_uiState.enableGravitationalRedshift ||
        lastShowDisk != m_uiState.showAccretionDisk) {
        
        // Update the simulation config to reflect UI settings
        BHErrorCode error = bh_configure_simulation(
            m_physicsContext,
            0.01,                          // Time step
            200.0,                         // Max ray distance
            std::max(300, m_uiState.integrationSteps), // Integration steps
            1e-7                           // Tolerance
        );
        
        // Additional settings that need explicit update
        SimulationConfig config;
        config.enable_doppler = m_uiState.enableDopplerEffect ? 1 : 0;
        config.enable_gravitational_redshift = m_uiState.enableGravitationalRedshift ? 1 : 0;
        config.show_accretion_disk = m_uiState.showAccretionDisk ? 1 : 0;
        
        // Set doppler factor to a stronger value (0.5) to make it more visible
        config.doppler_factor = 0.5;
        
        // Apply these settings to the physics context
        // Note: In a real implementation, there should be an API call to set these
        // directly, but for now we're working with what we have
        
        lastDopplerEffect = m_uiState.enableDopplerEffect;
        lastRedshift = m_uiState.enableGravitationalRedshift;
        lastShowDisk = m_uiState.showAccretionDisk;
        parametersChanged = true;
    }
    
    // If any parameters changed, update the ray trace texture
    if (parametersChanged) {
        updateRayTraceTexture();
    }
}

void Renderer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    // Get the renderer instance
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    
    // Update the viewport
    glViewport(0, 0, width, height);
    
    // Update the renderer's width and height
    renderer->m_width = width;
    renderer->m_height = height;
}

void Renderer::renderAccretionDiskParticles() {
    if (!m_uiState.showAccretionDisk) {
        return;
    }

    // Create particles with full GR orbital mechanics
    const int numParticles = 500;
    
    // Calculate the black hole Schwarzschild radius (r_s = 2GM/c²)
    float schwarzschildRadius = 2.0f * m_uiState.mass;
    
    // Calculate spin parameter (a/M) - dimensionless
    float spin = m_uiState.spin;
    
    // Inner radius depends on spin (ISCO - innermost stable circular orbit)
    // For Schwarzschild (spin=0): ISCO = 6M
    // For extreme Kerr (spin=1): ISCO = 1M (prograde), 9M (retrograde)
    // This is a simple approximation of the ISCO formula
    float innerRadius = 0.0f;
    if (spin < 0.001f) {
        // Schwarzschild case
        innerRadius = 6.0f * m_uiState.mass;
    } else {
        // Kerr case (simplified formula for prograde orbits)
        // r_ISCO ≈ 3 + Z₂ - √((3-Z₁)(3+Z₁+2Z₂))
        // where Z₁ = 1+(1-a²)^(1/3)*((1+a)^(1/3)+(1-a)^(1/3))
        // and Z₂ = √(3a² + Z₁²)
        // This is simplified to an approximation:
        innerRadius = (6.0f - 5.2f * spin + 1.2f * spin * spin) * m_uiState.mass;
    }
    
    // Outer radius of the disk
    float outerRadius = 30.0f * m_uiState.mass;
    
    // Reserve memory for particle data with size caps to prevent overflow
    const int MAX_PARTICLES = 5000; // Reasonable upper limit
    int safeNumParticles = std::min(numParticles, MAX_PARTICLES);
    
    std::vector<float> positions;
    std::vector<float> velocities;
    std::vector<int> types(safeNumParticles);
    
    positions.reserve(safeNumParticles * 3);
    velocities.reserve(safeNumParticles * 3);
    
    // Generate particles in a disk with proper GR orbital velocities
    for (int i = 0; i < safeNumParticles; i++) {
        // Use square root distribution for uniform density
        float r = innerRadius + (outerRadius - innerRadius) * sqrt(static_cast<float>(i) / safeNumParticles);
        
        // Random angle
        float angle = static_cast<float>(rand()) / RAND_MAX * 2.0f * M_PI;
        
        // Disk thickness increases with radius (h/r ~ 0.1)
        float thickness = 0.1f * r;
        float z = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * thickness;
        
        // Position in cylindrical coordinates (r, θ, z)
        float x = r * cos(angle);
        float y = z;
        float z_pos = r * sin(angle);
        
        // Add position to vector
        positions.push_back(x);
        positions.push_back(y);
        positions.push_back(z_pos);
        
        // Calculate GR orbital velocity
        // For Schwarzschild metric: v_φ = sqrt(M/r) / sqrt(1 - 2M/r)
        // For Kerr metric (simplified): v_φ = sqrt(M/r³) * (r^(3/2) + a) / sqrt(r³ - 3Mr² + 2aM√r)
        
        float v_phi = 0.0f;
        
        if (spin < 0.001f) {
            // Schwarzschild case
            v_phi = sqrt(m_uiState.mass / r) / sqrt(std::max(0.1f, 1.0f - 2.0f * m_uiState.mass / r));
        } else {
            // Kerr metric (simplified for prograde equatorial orbits)
            float r32 = pow(r, 1.5f);
            float num = sqrt(m_uiState.mass / (r*r*r)) * (r32 + spin * m_uiState.mass);
            float denom = sqrt(std::max(0.1f, r*r*r - 3.0f * m_uiState.mass * r*r + 2.0f * spin * m_uiState.mass * r32));
            v_phi = num / denom;
        }
        
        // Tangential velocity vector (-sin θ, 0, cos θ)
        float vx = -v_phi * sin(angle);
        float vy = 0.0f;
        float vz = v_phi * cos(angle);
        
        // Add velocity to vector
        velocities.push_back(vx);
        velocities.push_back(vy);
        velocities.push_back(vz);
        
        // All particles are type 0 (accretion disk)
        types[i] = 0;
    }
    
    // Update particle data with thread safety
    {
        std::lock_guard<std::mutex> lock(m_renderDataMutex);
        m_particleData.positions = positions;
        m_particleData.velocities = velocities;
        m_particleData.types = types;
        m_particleData.count = safeNumParticles;
    }
    
    // Log update once every 500 calls to reduce console spam
    static int updateCount = 0;
    if (updateCount++ % 500 == 0) {
        printf("Updated particle data: %d particles\n", safeNumParticles);
    }
}

// Initialize the ray trace texture
bool Renderer::initRayTraceTexture() {
    // Set default resolution based on window size
    m_rtWidth = m_width;
    m_rtHeight = m_height;
    
    // Allocate memory for ray trace data buffer (RGBA)
    m_rayTraceData = new unsigned char[m_rtWidth * m_rtHeight * 4];
    if (!m_rayTraceData) {
        printf("Failed to allocate memory for ray trace data buffer\n");
        return false;
    }
    
    // Initialize with clear color (dark blue for space)
    memset(m_rayTraceData, 0, m_rtWidth * m_rtHeight * 4);
    for (int i = 0; i < m_rtWidth * m_rtHeight; i++) {
        m_rayTraceData[i*4+0] = 0;     // R
        m_rayTraceData[i*4+1] = 0;     // G
        m_rayTraceData[i*4+2] = 20;    // B
        m_rayTraceData[i*4+3] = 255;   // A
    }
    
    // Create OpenGL texture for ray trace results
    glGenTextures(1, &m_rayTraceTextureID);
    glBindTexture(GL_TEXTURE_2D, m_rayTraceTextureID);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Allocate texture storage
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_rtWidth, m_rtHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_rayTraceData);
    
    // Unbind texture
    glBindTexture(GL_TEXTURE_2D, 0);
    
    printf("Ray trace texture initialized: %dx%d\n", m_rtWidth, m_rtHeight);
    return true;
}

// Update the ray trace texture using backend physics
void Renderer::updateRayTraceTexture() {
    // Use double-buffering via progressive quality improvement
    static bool isFirstRender = true;
    static int frameCount = 0;
    static int updateFrequency = 1; // How often to update (every N frames)
    static int currentQuality = 32; // Track current quality level
    
    frameCount++;
    
    // Only update every N frames after initial renders
    if (!isFirstRender && frameCount % updateFrequency != 0) {
        return; // Skip this frame
    }
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // EMERGENCY FIX: Extreme performance optimization
    // Start with extremely low quality and improve very gradually
    int quality = currentQuality;
    int maxSteps = 50; // Default max integration steps
    
    // Progressive quality improvement - MUCH more conservative
    if (isFirstRender) {
        quality = 32; // Start with 1/32 resolution (tiny)
        currentQuality = quality;
        maxSteps = 20; // Use very few steps
        updateFrequency = 10; // Update very infrequently at first
    } else if (frameCount < 120) { // First 10 seconds at 12 FPS
        quality = 16; // 1/16 resolution
        currentQuality = quality;
        maxSteps = 30;
    } else if (frameCount < 240) { // Next 10 seconds
        quality = 8; // Then 1/8 resolution
        currentQuality = quality;
        maxSteps = 30;
        updateFrequency = 5; // Update more frequently
    } else if (frameCount < 480) { // Next 20 seconds
        quality = 4; // Then 1/4 resolution
        currentQuality = quality;
        maxSteps = 40;
        updateFrequency = 3; 
    } else {
        quality = 2; // Then 1/2 resolution - never go to full res
        currentQuality = quality;
        maxSteps = 50;
        updateFrequency = 2; // Update every other frame
    }
    
    // Calculate the observer position (camera)
    Vector3D cameraPos = m_camera->position;
    Vector3D forward = vector3D_normalize(vector3D_sub(m_camera->target, cameraPos));
    Vector3D right = vector3D_cross((Vector3D){0, 1, 0}, forward);
    float rightLen = vector3D_length(right);
    
    // Handle case where camera is looking straight up/down
    if (rightLen < 0.0001f) {
        right = (Vector3D){1, 0, 0};
    } else {
        right = vector3D_scale(right, 1.0f/rightLen);
    }
    
    Vector3D up = vector3D_cross(forward, right);
    float upLen = vector3D_length(up);
    up = vector3D_scale(up, 1.0f/upLen);
    
    // Calculate observer distance from black hole center
    double observerDistance = vector3D_length(cameraPos);
    
    // Reduced resolution for rendering
    int rtWidth = m_rtWidth / quality;
    int rtHeight = m_rtHeight / quality;
    
    // Don't start with 0 pixels
    rtWidth = std::max(rtWidth, 32);
    rtHeight = std::max(rtHeight, 18);
    
    // Generate a starfield background first
    for (int i = 0; i < m_rtWidth * m_rtHeight; i++) {
        // Default space color (very dark blue)
        m_rayTraceData[i*4+0] = 0;     // R
        m_rayTraceData[i*4+1] = 0;     // G
        m_rayTraceData[i*4+2] = 20;    // B
        m_rayTraceData[i*4+3] = 255;   // A
        
        // Add stars based on a simple hash
        unsigned int seed = i * 12347 + static_cast<int>(glfwGetTime() * 100) % 10;
        float r = static_cast<float>(seed % 1000) / 1000.0f;
        
        if (r > 0.998f) {
            // Bright star
            unsigned char brightness = 200 + (seed % 55);
            m_rayTraceData[i*4+0] = brightness;
            m_rayTraceData[i*4+1] = brightness;
            m_rayTraceData[i*4+2] = brightness;
        } 
        else if (r > 0.995f) {
            // Dim star with slight color
            unsigned char brightness = 100 + (seed % 155);
            m_rayTraceData[i*4+0] = brightness - 20;
            m_rayTraceData[i*4+1] = brightness;
            m_rayTraceData[i*4+2] = brightness + 30;
        }
    }
    
    // Use our enhanced fallback visualization instead of GPU ray tracing
    drawBlackHoleOverlay();
    
    // Update OpenGL texture
    glBindTexture(GL_TEXTURE_2D, m_rayTraceTextureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_rtWidth, m_rtHeight, GL_RGBA, GL_UNSIGNED_BYTE, m_rayTraceData);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    // Update first render flag
    isFirstRender = false;
    
    // Log performance info every 30 frames
    static int perfLogCount = 0;
    if (++perfLogCount % 30 == 0) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto renderTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - startTime);
        printf("Ray trace performance: %lld ms, quality: 1/%d, steps: %d\n", 
               renderTime.count(), quality, maxSteps);
    }

    // Make sure UI settings are properly passed to shader parameters
    // Add the following to updateRayTraceTexture at the end before isFirstRender = false
    // We're using the fake draw function, but in a real implementation this would
    // update shader uniforms
    if (m_blackHoleShader) {
        m_blackHoleShader->use();
        m_blackHoleShader->setInt("enableDoppler", m_uiState.enableDopplerEffect ? 1 : 0);
        m_blackHoleShader->setInt("enableRedshift", m_uiState.enableGravitationalRedshift ? 1 : 0);
        m_blackHoleShader->setInt("showDisk", m_uiState.showAccretionDisk ? 1 : 0);
    }
}

// Clean up ray trace resources
void Renderer::cleanupRayTrace() {
    // Delete OpenGL texture if it exists
    if (m_rayTraceTextureID) {
        glDeleteTextures(1, &m_rayTraceTextureID);
        m_rayTraceTextureID = 0;
    }
    
    // Free ray trace data memory
    if (m_rayTraceData) {
        delete[] m_rayTraceData;
        m_rayTraceData = nullptr;
    }
    
    m_rtWidth = 0;
    m_rtHeight = 0;
}

void Renderer::createFullScreenQuad() {
    float quadVertices[] = {
        // positions   // texture coords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    GLuint VBO;
    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(m_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    
    // Note: we don't delete VBO here because it's still used by the VAO
    glBindVertexArray(0);
}

void Renderer::cleanupBuffers() {
    if (m_quadVAO != 0) {
        glDeleteVertexArrays(1, &m_quadVAO);
        m_quadVAO = 0;
    }
    
    // Add any other buffer cleanup here
}

void Renderer::cleanupShaders() {
    if (m_blackHoleShader) {
        glDeleteProgram(m_blackHoleShader->program);
        delete m_blackHoleShader;
        m_blackHoleShader = nullptr;
    }
    
    if (m_accretionDiskShader) {
        glDeleteProgram(m_accretionDiskShader->program);
        delete m_accretionDiskShader;
        m_accretionDiskShader = nullptr;
    }
    
    if (m_skyboxShader) {
        glDeleteProgram(m_skyboxShader->program);
        delete m_skyboxShader;
        m_skyboxShader = nullptr;
    }
    
    if (m_particleShaderProgram != 0) {
        glDeleteProgram(m_particleShaderProgram);
        m_particleShaderProgram = 0;
    }
}

void Renderer::drawBlackHoleOverlay() {
    // Calculate the screen-space position of the black hole center
    int centerX = m_rtWidth / 2;
    int centerY = m_rtHeight / 2;
    
    // Calculate radius based on mass and a more scientifically accurate scale
    // Schwarzschild radius is 2GM/c² but we're using units where c=G=1, so r_s = 2M
    float schwarzschildRadius = 2.0f * m_uiState.mass;
    
    // For visualization, we'll make the black hole reasonably sized on screen
    float pixelsPerUnit = m_rtWidth / 40.0f;  // Adjust for better visibility
    float apparentSize = schwarzschildRadius * pixelsPerUnit;
    
    // Set minimum size for visibility
    float radius = std::max(apparentSize, 40.0f);
    
    // Clear the background with stars first
    for (int y = 0; y < m_rtHeight; y++) {
        for (int x = 0; x < m_rtWidth; x++) {
            int index = (y * m_rtWidth + x) * 4;
            
            // Default dark space color
            m_rayTraceData[index+0] = 0;      // R
            m_rayTraceData[index+1] = 0;      // G
            m_rayTraceData[index+2] = 20;     // B
            m_rayTraceData[index+3] = 255;    // A
            
            // Add stars based on a simple hash
            unsigned int seed = x * 12347 + y * 43112 + static_cast<int>(glfwGetTime() * 10) % 10;
            float r = static_cast<float>(seed % 1000) / 1000.0f;
            
            if (r > 0.997f) {
                // Bright star
                unsigned char brightness = 200 + (seed % 55);
                m_rayTraceData[index+0] = brightness;
                m_rayTraceData[index+1] = brightness;
                m_rayTraceData[index+2] = brightness;
            } 
            else if (r > 0.994f) {
                // Dim star with slight color
                unsigned char brightness = 100 + (seed % 155);
                m_rayTraceData[index+0] = brightness - 20;
                m_rayTraceData[index+1] = brightness;
                m_rayTraceData[index+2] = brightness + 30;
            }
        }
    }
    
    // Draw the black hole and accretion disk with improved visualization
    for (int y = 0; y < m_rtHeight; y++) {
        for (int x = 0; x < m_rtWidth; x++) {
            float dx = x - centerX;
            float dy = y - centerY;
            float dist = sqrt(dx*dx + dy*dy);
            
            int index = (y * m_rtWidth + x) * 4;
            
            // Accretion disk (with correct inner and outer radius)
            float innerDiskRadius = 3.0f * schwarzschildRadius; // Innermost stable circular orbit is at 3Rs for spin=0
            float outerDiskRadius = 15.0f * schwarzschildRadius;
            
            float diskInnerPixels = innerDiskRadius * pixelsPerUnit;
            float diskOuterPixels = outerDiskRadius * pixelsPerUnit;
            
            // Calculate angle for position in the disk
                float angle = atan2(dy, dx);
            
            // Simulate gravitational lensing effect for background stars
            // Stars closer to the black hole get distorted
            if (dist > radius * 1.1f && dist < diskOuterPixels * 1.5f) {
                // Calculate distortion factor
                float distortion = 0.0f;
                if (dist < radius * 5.0f) {
                    // Strong distortion near the black hole
                    distortion = 10.0f * (radius / dist);
                } else {
                    // Weaker distortion farther away
                    distortion = 2.0f * (radius / dist);
                }
                
                // Apply distortion by sampling from a different position
                int srcX = static_cast<int>(x - dx * distortion / 8.0f);
                int srcY = static_cast<int>(y - dy * distortion / 8.0f);
                
                // Ensure source coordinates are in bounds
                if (srcX >= 0 && srcX < m_rtWidth && srcY >= 0 && srcY < m_rtHeight) {
                    int srcIndex = (srcY * m_rtWidth + srcX) * 4;
                    
                    // Only apply distortion to actual stars (bright pixels)
                    if (m_rayTraceData[srcIndex] > 50 || m_rayTraceData[srcIndex+1] > 50 || m_rayTraceData[srcIndex+2] > 50) {
                        // Make distorted stars more intense
                        m_rayTraceData[index+0] = static_cast<unsigned char>(std::min(255, m_rayTraceData[srcIndex+0] + 20));
                        m_rayTraceData[index+1] = static_cast<unsigned char>(std::min(255, m_rayTraceData[srcIndex+1] + 20));
                        m_rayTraceData[index+2] = static_cast<unsigned char>(std::min(255, m_rayTraceData[srcIndex+2] + 20));
                    }
                }
            }

            if (m_uiState.showAccretionDisk && dist > diskInnerPixels && dist < diskOuterPixels) {
                // Apply warping to the disk to simulate frame dragging effect if spin is non-zero
                float warpedAngle = angle;
                if (abs(m_uiState.spin) > 0.001f) {
                    float warpFactor = m_uiState.spin * (diskOuterPixels - dist) / (diskOuterPixels - diskInnerPixels);
                    warpedAngle = angle + warpFactor * 0.5f;
                }
                
                // Apply Doppler shift - brighter/bluer on approaching side (left), dimmer/redder on receding side
                float dopplerFactor = m_uiState.enableDopplerEffect ? 
                                    (1.0f + 0.8f * cos(warpedAngle)) : 1.0f;  // Increased factor from 0.5 to 0.8
                
                // Temperature decreases with radius (T ∝ r^(-3/4) for thin disk)
                float relativeRadius = dist / diskInnerPixels;
                float temperature = pow(relativeRadius, -0.75f);
                temperature = std::min(temperature, 1.0f); // Clamp to max 1.0
                
                // Distance falloff
                float distanceFactor = 1.0f - ((dist - diskInnerPixels) / (diskOuterPixels - diskInnerPixels));
                distanceFactor = std::max(0.0f, distanceFactor);
                
                // Apply brightness based on accretion rate
                float brightness = m_uiState.accretionRate * dopplerFactor;
                
                // Add variation based on angle to create spiral arm effect
                float spiralFactor = 0.3f * (1.0f + sin(8.0f * warpedAngle + 5.0f * relativeRadius));
                brightness *= (0.7f + spiralFactor);
                
                // Map temperature to color (red->yellow->white)
                if (temperature < 0.5f) {
                    // Red to yellow (cooler outer regions)
                    m_rayTraceData[index+0] = static_cast<unsigned char>(255 * brightness);
                    m_rayTraceData[index+1] = static_cast<unsigned char>(255 * brightness * temperature * 2.0f);
                    m_rayTraceData[index+2] = 0;
                } else {
                    // Yellow to white-blue (hotter inner regions)
                    m_rayTraceData[index+0] = static_cast<unsigned char>(255 * brightness);
                    m_rayTraceData[index+1] = static_cast<unsigned char>(255 * brightness);
                    m_rayTraceData[index+2] = static_cast<unsigned char>(255 * brightness * (temperature - 0.5f) * 2.0f);
                }
                
                // Apply gravitational redshift if enabled
                if (m_uiState.enableGravitationalRedshift) {
                    // Simplified gravitational redshift factor: sqrt(1 - 2M/r)
                    float rInM = relativeRadius * innerDiskRadius / m_uiState.mass;
                    float redshiftFactor = sqrt(std::max(0.1f, 1.0f - 2.0f/rInM));
                    
                    // Reduce blue component more than red (redshifting)
                    m_rayTraceData[index+2] = static_cast<unsigned char>(m_rayTraceData[index+2] * redshiftFactor);
                    m_rayTraceData[index+1] = static_cast<unsigned char>(m_rayTraceData[index+1] * (redshiftFactor + 1.0f) * 0.5f);
                }
            }
            
            // Blue glow around the black hole (gravitational lensing effect)
            float lensRegionOuter = radius * 1.5f;
            float lensRegionInner = radius * 1.05f;
            
            if (dist < lensRegionOuter && dist > lensRegionInner) {
                float intensity = 1.0f - (dist - lensRegionInner) / (lensRegionOuter - lensRegionInner);
                intensity = intensity * intensity; // Square for sharper falloff
                
                // Blue-purple glow for lensing
                m_rayTraceData[index+0] = static_cast<unsigned char>(std::min(255.0f, m_rayTraceData[index+0] + 40.0f * intensity));
                m_rayTraceData[index+1] = static_cast<unsigned char>(std::min(255.0f, m_rayTraceData[index+1] + 80.0f * intensity));
                m_rayTraceData[index+2] = static_cast<unsigned char>(std::min(255.0f, m_rayTraceData[index+2] + 200.0f * intensity));
            }
            
            // Black event horizon
            if (dist < radius) {
                m_rayTraceData[index+0] = 0;
                m_rayTraceData[index+1] = 0;
                m_rayTraceData[index+2] = 0;
            }
            
            // Add a thin bright ring at event horizon edge for definition
            float ringWidth = radius * 0.05f;
            if (dist > radius - ringWidth && dist < radius + ringWidth) {
                float intensity = 1.0f - abs(dist - radius) / ringWidth;
                intensity = intensity * intensity; // Square for sharper edge
                
                // Blue-white ring
                m_rayTraceData[index+0] = static_cast<unsigned char>(std::min(255.0f, m_rayTraceData[index+0] + 80.0f * intensity));
                m_rayTraceData[index+1] = static_cast<unsigned char>(std::min(255.0f, m_rayTraceData[index+1] + 120.0f * intensity));
                m_rayTraceData[index+2] = static_cast<unsigned char>(std::min(255.0f, m_rayTraceData[index+2] + 200.0f * intensity));
            }
            
            // Add a faint Einstein ring (light from behind the black hole bent around it)
            float einsteinRadius = radius * 2.5f;
            float einsteinWidth = radius * 0.2f;
            if (abs(dist - einsteinRadius) < einsteinWidth) {
                float intensity = 1.0f - abs(dist - einsteinRadius) / einsteinWidth;
                intensity = intensity * intensity * 0.7f; // Adjust brightness
                
                // Golden-white Einstein ring
                m_rayTraceData[index+0] = static_cast<unsigned char>(std::min(255.0f, m_rayTraceData[index+0] + 200.0f * intensity));
                m_rayTraceData[index+1] = static_cast<unsigned char>(std::min(255.0f, m_rayTraceData[index+1] + 180.0f * intensity));
                m_rayTraceData[index+2] = static_cast<unsigned char>(std::min(255.0f, m_rayTraceData[index+2] + 100.0f * intensity));
            }
        }
    }
}

bool Renderer::initTemporalBuffer(int width, int height) {
    // Free old buffers if they exist
    if (m_temporalBuffer.accumulated_color) {
        delete[] m_temporalBuffer.accumulated_color;
    }
    if (m_temporalBuffer.history_color) {
        delete[] m_temporalBuffer.history_color;
    }
    if (m_edgeFactorBuffer) {
        delete[] m_edgeFactorBuffer;
    }
    
    // Allocate new buffers
    const size_t buffer_size = width * height * 3 * sizeof(float);
    m_temporalBuffer.accumulated_color = new float[width * height * 3];
    m_temporalBuffer.history_color = new float[width * height * 3];
    m_edgeFactorBuffer = new float[width * height];
    
    // Check if allocation succeeded
    if (!m_temporalBuffer.accumulated_color || !m_temporalBuffer.history_color || !m_edgeFactorBuffer) {
        // Clean up on failure
        if (m_temporalBuffer.accumulated_color) {
            delete[] m_temporalBuffer.accumulated_color;
            m_temporalBuffer.accumulated_color = nullptr;
        }
        if (m_temporalBuffer.history_color) {
            delete[] m_temporalBuffer.history_color;
            m_temporalBuffer.history_color = nullptr;
        }
        if (m_edgeFactorBuffer) {
            delete[] m_edgeFactorBuffer;
            m_edgeFactorBuffer = nullptr;
        }
        return false;
    }
    
    // Initialize buffers to zero
    memset(m_temporalBuffer.accumulated_color, 0, buffer_size);
    memset(m_temporalBuffer.history_color, 0, buffer_size);
    memset(m_edgeFactorBuffer, 0, width * height * sizeof(float));
    
    // Set up buffer parameters
    m_temporalBuffer.width = width;
    m_temporalBuffer.height = height;
    m_temporalBuffer.frame_count = 0;
    m_temporalBuffer.blend_factor = 0.1f;
    m_temporalBuffer.reset_accumulation = true;
    m_temporalBuffer.max_frames = 32;
    m_temporalBuffer.jitter_index = 0;
    
    return true;
}

void Renderer::resetAccumulation() {
    if (!m_temporalBuffer.accumulated_color || !m_temporalBuffer.history_color) {
        return;
    }
    
    // Clear accumulation buffers
    const size_t buffer_size = m_temporalBuffer.width * m_temporalBuffer.height * 3 * sizeof(float);
    memset(m_temporalBuffer.accumulated_color, 0, buffer_size);
    memset(m_temporalBuffer.history_color, 0, buffer_size);
    
    // Reset frame count
    m_temporalBuffer.frame_count = 0;
    m_temporalBuffer.reset_accumulation = false;
}

void Renderer::accumulateFrame(const float* newFrame) {
    if (!m_temporalBuffer.accumulated_color || !m_temporalBuffer.history_color || !newFrame) {
        return;
    }
    
    // Check if we need to reset accumulation
    if (m_temporalBuffer.reset_accumulation) {
        resetAccumulation();
    }
    
    // Calculate blend factor based on frame count
    // Use higher weight for initial frames, then gradually decrease
    float alpha;
    if (m_temporalBuffer.frame_count == 0) {
        // First frame gets full weight
        alpha = 1.0f;
    } else if (m_temporalBuffer.frame_count < 5) {
        // First few frames get higher weight to establish image quickly
        alpha = 0.5f;
    } else {
        // Later frames get progressively lower weight
        // This helps stabilize the image while still allowing for updates
        alpha = m_temporalBuffer.blend_factor;
    }
    
    // Store current accumulation buffer in history
    memcpy(m_temporalBuffer.history_color, m_temporalBuffer.accumulated_color, 
           m_temporalBuffer.width * m_temporalBuffer.height * 3 * sizeof(float));
    
    // Blend new frame with accumulated buffer
    const int total_pixels = m_temporalBuffer.width * m_temporalBuffer.height * 3;
    for (int i = 0; i < total_pixels; i++) {
        m_temporalBuffer.accumulated_color[i] = 
            m_temporalBuffer.accumulated_color[i] * (1.0f - alpha) + newFrame[i] * alpha;
    }
    
    // Update frame count (capped at max_frames)
    m_temporalBuffer.frame_count = std::min(m_temporalBuffer.frame_count + 1, m_temporalBuffer.max_frames);
    
    // Update jitter index for next frame
    m_temporalBuffer.jitter_index = (m_temporalBuffer.jitter_index + 1) % 64;
}

void Renderer::detectEdges() {
    if (!m_temporalBuffer.accumulated_color || !m_edgeFactorBuffer) {
        return;
    }
    
    const int width = m_temporalBuffer.width;
    const int height = m_temporalBuffer.height;
    
    // Detect edges by looking at color gradients
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            // Get center pixel color
            const float* center = &m_temporalBuffer.accumulated_color[(y * width + x) * 3];
            
            // Calculate max gradient with neighboring pixels
            float max_gradient = 0.0f;
            
            // Check 4 adjacent neighbors (up, down, left, right)
            const int dx[4] = {0, 0, -1, 1};
            const int dy[4] = {-1, 1, 0, 0};
            
            for (int i = 0; i < 4; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                
                const float* neighbor = &m_temporalBuffer.accumulated_color[(ny * width + nx) * 3];
                
                // Calculate color difference
                float gradient = 0.0f;
                for (int c = 0; c < 3; c++) {
                    gradient += fabs(center[c] - neighbor[c]);
                }
                gradient /= 3.0f;
                
                if (gradient > max_gradient) {
                    max_gradient = gradient;
                }
            }
            
            // Edge threshold - this would be configurable
            const float threshold = 0.1f;
            
            // If gradient exceeds threshold, mark as edge
            if (max_gradient > threshold) {
                m_edgeFactorBuffer[y * width + x] = 1.0f;
            } else {
                // Gradually reduce edge factor for pixels that are no longer edges
                m_edgeFactorBuffer[y * width + x] = std::max(0.0f, m_edgeFactorBuffer[y * width + x] - 0.2f);
            }
        }
    }
}

void Renderer::finalizeAccumulatedFrame() {
    if (!m_temporalBuffer.accumulated_color || !m_rayTraceData) {
        return;
    }
    
    // Apply any post-processing here if needed
    
    // Convert floating-point RGB to 8-bit RGBA
    const int width = m_temporalBuffer.width;
    const int height = m_temporalBuffer.height;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 4;
            for (int c = 0; c < 3; c++) {
                float color = m_temporalBuffer.accumulated_color[index + c];
                color = pow(std::max(0.0f, std::min(1.0f, color)), 1.0f / 2.2f);
                m_rayTraceData[index + c] = static_cast<unsigned char>(color * 255.0f);
            }
            m_rayTraceData[index + 3] = 255;
        }
    }
}

// Fix the initComputeResources function to properly return
bool Renderer::initComputeResources() {
    // First check if compute shaders are available by trying to load the functions
    if (!loadComputeShaderFunctions()) {
        printf("Compute shaders not available, GPU raytracing disabled\n");
        m_computeResources.initialized = false;
        m_uiState.useGPUAcceleration = false;
        return false;
    }
    
    // Set initial state
    m_computeResources.initialized = false;
    m_computeResources.width = m_rtWidth;
    m_computeResources.height = m_rtHeight;
    m_computeResources.workGroupSize = 16; // Default work group size
    
    // 1. Load and compile the compute shader
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    if (!computeShader) {
        printf("Failed to create compute shader object\n");
        return false;
    }
    
    // Read shader source from file
    std::string shaderSource = readShaderFile("src/visualization/shaders/ray_tracer.comp");
    if (shaderSource.empty()) {
        printf("Failed to read compute shader source\n");
        glDeleteShader(computeShader);
        return false;
    }
    
    // Compile shader
    const char* source = shaderSource.c_str();
    glShaderSource(computeShader, 1, &source, NULL);
    glCompileShader(computeShader);
    
    // Check for errors
    GLint success;
    char infoLog[512];
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(computeShader, 512, NULL, infoLog);
        printf("Compute shader compilation failed: %s\n", infoLog);
        glDeleteShader(computeShader);
        return false;
    }
    
    // 2. Create program and link
    m_computeResources.computeProgram = glCreateProgram();
    glAttachShader(m_computeResources.computeProgram, computeShader);
    glLinkProgram(m_computeResources.computeProgram);
    
    // Check for errors
    glGetProgramiv(m_computeResources.computeProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_computeResources.computeProgram, 512, NULL, infoLog);
        printf("Compute shader program linking failed: %s\n", infoLog);
        glDeleteShader(computeShader);
        glDeleteProgram(m_computeResources.computeProgram);
        m_computeResources.computeProgram = 0;
        return false;
    }
    
    // Clean up shader now that we've linked it
    glDeleteShader(computeShader);
    
    // 3. Create buffers
    
    // Ray input buffer (origins and directions)
    glGenBuffers(1, &m_computeResources.rayInputSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_computeResources.rayInputSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 
                 m_rtWidth * m_rtHeight * sizeof(GLfloat) * 8, // 2 vec4s per ray
                 NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    // Ray output buffer (colors)
    glGenBuffers(1, &m_computeResources.rayOutputSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_computeResources.rayOutputSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 
                 m_rtWidth * m_rtHeight * sizeof(GLfloat) * 4, // RGBA per pixel
                 NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    // Black hole parameters uniform buffer
    glGenBuffers(1, &m_computeResources.blackHoleParamsUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, m_computeResources.blackHoleParamsUBO);
    glBufferData(GL_UNIFORM_BUFFER, 64 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    // Set up workgroup size
    GLint maxWorkGroupSize[3];
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxWorkGroupSize[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxWorkGroupSize[1]);
    
    // Use a reasonable work group size (16x16 is common)
    m_computeResources.workGroupSize = std::min(16, std::min(maxWorkGroupSize[0], maxWorkGroupSize[1]));
    
    // Mark as initialized
    m_computeResources.initialized = true;
    m_uiState.useGPUAcceleration = true;
    
    printf("Compute shader resources initialized successfully. Work group size: %d\n", 
           m_computeResources.workGroupSize);
    
    // Immediately update parameters
    updateComputeResources();
    
    return true;
}

void Renderer::updateComputeResources() {
    if (!m_computeResources.initialized || !m_physicsContext) {
        return;
    }
    
    // Check if dimensions have changed
    if (m_computeResources.width != m_rtWidth || m_computeResources.height != m_rtHeight) {
        // Resize buffers
        cleanupComputeResources();
        initComputeResources();
        return;
    }
    
    // Update uniform buffer with black hole parameters
    glBindBuffer(GL_UNIFORM_BUFFER, m_computeResources.blackHoleParamsUBO);
    
    // Collect parameters
    float params[64]; // Large enough for all parameters
    memset(params, 0, sizeof(params)); // Initialize to zero
    
    // We're skipping calling bh_generate_shader_data for now to avoid linker errors
    // Just populate with some default values
    params[0] = 1.0f;  // mass
    params[1] = 0.0f;  // spin
    params[2] = 2.0f;  // schwarzschild_radius
    
    // Update buffer
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(params), params);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    // Update ray input buffer with ray origins and directions
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_computeResources.rayInputSSBO);
    
    // Generate ray data for each pixel
    const size_t rayBufferSize = m_rtWidth * m_rtHeight * sizeof(GLfloat) * 4 * 2;
    GLfloat* rayData = new GLfloat[m_rtWidth * m_rtHeight * 8]; // 2 vec4s per ray
    
    if (!rayData) {
        printf("Failed to allocate memory for ray data\n");
        return;
    }
    
    // Camera parameters
    Vector3D target_minus_position = vector3D_sub(m_camera->target, m_camera->position);
    Vector3D forward = vector3D_normalize(target_minus_position);
    Vector3D right = vector3D_cross(forward, (Vector3D){0, 1, 0});
    right = vector3D_normalize(right);
    Vector3D up = vector3D_cross(right, forward);
    
    // FOV and aspect ratio
    float aspect_ratio = (float)m_rtWidth / (float)m_rtHeight;
    float tan_half_fov = tan(m_camera->fov * BH_PI / 180.0f * 0.5f);
    
    // Generate ray for each pixel
    for (int y = 0; y < m_rtHeight; y++) {
        for (int x = 0; x < m_rtWidth; x++) {
            int idx = (y * m_rtWidth + x) * 8; // 8 floats per ray (2 vec4s)
            
            // Convert pixel coordinates to normalized device coordinates
            float px = (2.0f * ((float)x + 0.5f) / m_rtWidth - 1.0f) * aspect_ratio * tan_half_fov;
            float py = (1.0f - 2.0f * ((float)y + 0.5f) / m_rtHeight) * tan_half_fov;
            
            // Ray origin (vec4)
            rayData[idx + 0] = m_camera->position.x;
            rayData[idx + 1] = m_camera->position.y;
            rayData[idx + 2] = m_camera->position.z;
            rayData[idx + 3] = 0.0f; // w component
            
            // Ray direction (vec4)
            Vector3D dir;
            dir.x = forward.x + px * right.x + py * up.x;
            dir.y = forward.y + px * right.y + py * up.y;
            dir.z = forward.z + px * right.z + py * up.z;
            
            // Normalize direction
            float len = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
            dir.x /= len;
            dir.y /= len;
            dir.z /= len;
            
            rayData[idx + 4] = dir.x;
            rayData[idx + 5] = dir.y;
            rayData[idx + 6] = dir.z;
            rayData[idx + 7] = 0.0f; // w component
        }
    }
    
    // Update buffer
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, rayBufferSize, rayData);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    // Clean up
    delete[] rayData;
}

// Modify executeRayTracingOnGPU to safely handle the case when compute shaders are not available
void Renderer::executeRayTracingOnGPU() {
    // Check if initialized
    if (!m_computeResources.initialized || !glDispatchCompute || !glMemoryBarrier) {
        // Fall back to CPU raytracing
        printf("Compute shaders not available, using CPU fallback\n");
        return;
    }
    
    // Update compute resources if needed 
    updateComputeResources();
    
    // Use the compute shader program
    glUseProgram(m_computeResources.computeProgram);
    
    // Bind resources
    
    // 1. Bind ray input buffer to binding point 0
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_computeResources.rayInputSSBO);
    
    // 2. Bind ray output buffer to binding point 1
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_computeResources.rayOutputSSBO);
    
    // 3. Bind uniform buffer with black hole parameters to binding point 0
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_computeResources.blackHoleParamsUBO);
    
    // 4. Set additional uniform values if needed
    GLuint maxStepsLocation = glGetUniformLocation(m_computeResources.computeProgram, "u_MaxSteps");
    if (maxStepsLocation != GL_INVALID_INDEX) {
        glUniform1i(maxStepsLocation, m_uiState.integrationSteps);
    }
    
    GLuint resolutionLocation = glGetUniformLocation(m_computeResources.computeProgram, "u_Resolution");
    if (resolutionLocation != GL_INVALID_INDEX) {
        glUniform2i(resolutionLocation, m_rtWidth, m_rtHeight);
    }
    
    // Calculate dispatch dimensions (how many work groups we need)
    int groupSizeX = (m_rtWidth + m_computeResources.workGroupSize - 1) / m_computeResources.workGroupSize;
    int groupSizeY = (m_rtHeight + m_computeResources.workGroupSize - 1) / m_computeResources.workGroupSize;
    
    // Dispatch compute shader
    glDispatchCompute(groupSizeX, groupSizeY, 1);
    
    // Wait for compute shader to finish
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    // Read back the ray output buffer to our ray trace data buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_computeResources.rayOutputSSBO);
    
    // Map buffer to client memory for reading
    float* gpuOutputData = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    
    if (gpuOutputData) {
        // Convert the float RGBA data (0.0-1.0) to unsigned bytes (0-255)
        for (int i = 0; i < m_rtWidth * m_rtHeight; i++) {
            m_rayTraceData[i*4+0] = static_cast<unsigned char>(std::min(1.0f, gpuOutputData[i*4+0]) * 255.0f);
            m_rayTraceData[i*4+1] = static_cast<unsigned char>(std::min(1.0f, gpuOutputData[i*4+1]) * 255.0f);
            m_rayTraceData[i*4+2] = static_cast<unsigned char>(std::min(1.0f, gpuOutputData[i*4+2]) * 255.0f);
            m_rayTraceData[i*4+3] = static_cast<unsigned char>(std::min(1.0f, gpuOutputData[i*4+3]) * 255.0f);
        }
        
        // Unmap buffer
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    } else {
        printf("Failed to map GPU output buffer\n");
    }
    
    // Unbind buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void Renderer::cleanupComputeResources() {
    if (!m_computeResources.initialized) {
        return;
    }
    
    // Delete program
    if (m_computeResources.computeProgram) {
        glDeleteProgram(m_computeResources.computeProgram);
        m_computeResources.computeProgram = 0;
    }
    
    // Delete buffers
    if (m_computeResources.rayInputSSBO) {
        glDeleteBuffers(1, &m_computeResources.rayInputSSBO);
        m_computeResources.rayInputSSBO = 0;
    }
    
    if (m_computeResources.rayOutputSSBO) {
        glDeleteBuffers(1, &m_computeResources.rayOutputSSBO);
        m_computeResources.rayOutputSSBO = 0;
    }
    
    if (m_computeResources.blackHoleParamsUBO) {
        glDeleteBuffers(1, &m_computeResources.blackHoleParamsUBO);
        m_computeResources.blackHoleParamsUBO = 0;
    }
    
    m_computeResources.initialized = false;
}

// Helper function to read shader file
std::string Renderer::readShaderFile(const std::string& filePath) {
    std::ifstream shaderFile(filePath);
    std::string shaderCode;
    
    if (!shaderFile.is_open()) {
        std::cerr << "Failed to open shader file: " << filePath << std::endl;
        return "";
    }
    
    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    shaderFile.close();
    shaderCode = shaderStream.str();
    
    return shaderCode;
} 