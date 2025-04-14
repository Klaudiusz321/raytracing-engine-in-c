#include "renderer.h"
#include <iostream>
#include <chrono>

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

// Fragment shader for black hole simulation
// This is a placeholder - the actual ray tracing shader would be much more complex
const char* blackHoleFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform float mass;
uniform float spin;
uniform vec3 cameraPos;
uniform int integrationSteps;
uniform bool enableDopplerEffect;
uniform bool enableGravitationalRedshift;

void main() {
    // This is just a placeholder
    // The actual shader would perform ray tracing through curved spacetime
    vec2 uv = TexCoord * 2.0 - 1.0;
    float dist = length(uv);
    
    // Simple approximation of black hole shadow
    float shadowRadius = 2.6 * mass;
    
    if (dist < shadowRadius) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black hole
    } else {
        // Simplified gravitational lensing effect
        vec3 color = vec3(0.1, 0.2, 0.5); // Background color
        
        // Add a simple distortion effect
        float lensing = 1.0 - shadowRadius / (dist * 2.0);
        color *= lensing;
        
        FragColor = vec4(color, 1.0);
    }
}
)";

Renderer::Renderer() :
    m_window(nullptr),
    m_width(800),
    m_height(600),
    m_physicsContext(nullptr),
    m_running(false),
    m_renderDataFront(nullptr),
    m_renderDataBack(nullptr),
    m_dataReady(false),
    m_blackHoleShader(nullptr),
    m_accretionDiskShader(nullptr),
    m_skyboxShader(nullptr),
    m_camera(nullptr)
{
    // Initialize UI state with default values
    m_uiState.mass = 1.0f;
    m_uiState.spin = 0.0f;
    m_uiState.accretionRate = 0.5f;
    m_uiState.showAccretionDisk = true;
    m_uiState.showGrid = false;
    m_uiState.showStars = true;
    m_uiState.enableDopplerEffect = true;
    m_uiState.enableGravitationalRedshift = true;
    m_uiState.integrationSteps = 100;
    m_uiState.timeScale = 1.0f;
}

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::initialize(int width, int height, const char* title) {
    m_width = width;
    m_height = height;
    
    // Initialize OpenGL
    if (!initOpenGL()) {
        std::cerr << "Failed to initialize OpenGL context" << std::endl;
        return false;
    }
    
    // Set the window title
    glfwSetWindowTitle(m_window, title);
    
    // Initialize ImGui
    initImGui();
    
    // Create render data structures
    m_renderDataFront = new RenderData();
    m_renderDataBack = new RenderData();
    
    // Initialize camera
    m_camera = new Camera();
    
    // Initialize shaders
    if (!initShaders()) {
        std::cerr << "Failed to initialize shaders" << std::endl;
        return false;
    }
    
    // Initialize physics engine
    if (!initPhysics()) {
        std::cerr << "Failed to initialize physics engine" << std::endl;
        return false;
    }
    
    // Start physics thread
    m_running = true;
    m_physicsThread = std::thread(&Renderer::physicsThreadFunc, this);
    
    return true;
}

void Renderer::runMainLoop() {
    // Delta time tracking
    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    
    // Main loop
    while (!glfwWindowShouldClose(m_window)) {
        // Calculate delta time
        auto currentFrameTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentFrameTime - lastFrameTime).count();
        lastFrameTime = currentFrameTime;
        
        // Poll for events
        glfwPollEvents();
        
        // Update camera based on input
        updateCamera(deltaTime);
        
        // Update physics parameters if UI changed
        updatePhysicsParams();
        
        // Render a frame
        renderFrame();
        
        // Render UI
        renderUI();
        
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
    
    // Cleanup physics engine
    if (m_physicsContext) {
        bh_shutdown(m_physicsContext);
        m_physicsContext = nullptr;
    }
    
    // Cleanup render data
    delete m_renderDataFront;
    delete m_renderDataBack;
    
    // Cleanup camera
    delete m_camera;
    
    // Cleanup shaders
    delete m_blackHoleShader;
    delete m_accretionDiskShader;
    delete m_skyboxShader;
    
    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // Cleanup GLFW
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
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create a window
    m_window = glfwCreateWindow(m_width, m_height, "Black Hole Renderer", NULL, NULL);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    // Make the window's context current
    glfwMakeContextCurrent(m_window);
    
    // Set framebuffer resize callback
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    
    // Initialize GLAD
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }
   \
    
    return true;
}

void Renderer::initImGui() {
    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup ImGui style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

bool Renderer::initShaders() {
    // Create black hole shader
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
        std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
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
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
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
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        return false;
    }
    
    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    // In a full implementation, we would also create accretion disk and skybox shaders here
    
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
    // Clear the framebuffer
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // In a full implementation, this would render:
    // 1. The skybox (stars, galaxies)
    // 2. The black hole (ray traced)
    // 3. The accretion disk (particles or textured quad)
    
    // For now, we just use a simple shader to demonstrate black hole shadow
    // This would be replaced with a much more sophisticated ray tracing shader
    
    // Set up a fullscreen quad for ray tracing
    static const float vertices[] = {
        // Positions   // Texture Coords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    
    // Generate VAO, VBO
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Use the black hole shader
    m_blackHoleShader->use();
    
    // Send uniforms to shader
    m_blackHoleShader->setFloat("mass", m_uiState.mass);
    m_blackHoleShader->setFloat("spin", m_uiState.spin);
    m_blackHoleShader->setVec3("cameraPos", 
                              m_camera->position.x, 
                              m_camera->position.y, 
                              m_camera->position.z);
    m_blackHoleShader->setInt("integrationSteps", m_uiState.integrationSteps);
    m_blackHoleShader->setInt("enableDopplerEffect", m_uiState.enableDopplerEffect ? 1 : 0);
    m_blackHoleShader->setInt("enableGravitationalRedshift", m_uiState.enableGravitationalRedshift ? 1 : 0);
    
    // Draw the quad
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    // Cleanup
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

void Renderer::renderUI() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Create a window for black hole parameters
    ImGui::Begin("Black Hole Parameters");
    
    // Black hole mass slider
    ImGui::SliderFloat("Mass (M)", &m_uiState.mass, 0.1f, 10.0f, "%.1f M");
    
    // Black hole spin slider
    ImGui::SliderFloat("Spin (a/M)", &m_uiState.spin, 0.0f, 0.998f, "%.3f");
    
    // Accretion rate slider
    ImGui::SliderFloat("Accretion Rate", &m_uiState.accretionRate, 0.0f, 1.0f, "%.2f");
    
    ImGui::Separator();
    
    // Visualization options
    ImGui::Checkbox("Show Accretion Disk", &m_uiState.showAccretionDisk);
    ImGui::Checkbox("Show Grid", &m_uiState.showGrid);
    ImGui::Checkbox("Show Stars", &m_uiState.showStars);
    
    ImGui::Separator();
    
    // Physics effects
    ImGui::Checkbox("Doppler Effect", &m_uiState.enableDopplerEffect);
    ImGui::Checkbox("Gravitational Redshift", &m_uiState.enableGravitationalRedshift);
    
    ImGui::Separator();
    
    // Integration steps slider
    ImGui::SliderInt("Integration Steps", &m_uiState.integrationSteps, 10, 1000);
    
    // Simulation speed slider
    ImGui::SliderFloat("Time Scale", &m_uiState.timeScale, 0.0f, 10.0f, "%.1fx");
    
    ImGui::Separator();
    
    // Display some simulation stats
    // In a full implementation, this would show real data from the physics engine
    ImGui::Text("Shadow Radius: %.2f M", 2.6 * m_uiState.mass);
    ImGui::Text("ISCO Radius: %.2f M", 6.0);
    ImGui::Text("Observer Position: (%.1f, %.1f, %.1f) M", 
                m_camera->position.x, m_camera->position.y, m_camera->position.z);
    
    // Get synchronized data from physics thread
    {
        std::lock_guard<std::mutex> lock(m_renderDataMutex);
        if (m_dataReady) {
            ImGui::Text("Simulation Time: %.2f M", m_renderDataFront->simulationTime);
            ImGui::Text("Particle Count: %zu", m_renderDataFront->particlePositions.size());
        }
    }
    
    ImGui::End();
    
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Renderer::updateCamera(float deltaTime) {
    // In a full implementation, this would handle keyboard/mouse input
    // to move the camera around the black hole
    
    // For now, just a simple automatic rotation
    static float angle = 0.0f;
    angle += deltaTime * 0.1f; // Slow rotation
    
    float distance = 30.0f;
    m_camera->position.x = distance * sin(angle);
    m_camera->position.z = distance * cos(angle);
    
    // Always look at the center (black hole position)
    m_camera->target = {0.0, 0.0, 0.0};
}

void Renderer::physicsThreadFunc() {
    // Create a particle system
    void* particleSystem = bh_create_particle_system(m_physicsContext, 1000);
    
    if (!particleSystem) {
        std::cerr << "Failed to create particle system" << std::endl;
        return;
    }
    
    // Create accretion disk particles
    int numParticles = bh_create_accretion_disk_particles(m_physicsContext, particleSystem, 1000);
    
    if (numParticles <= 0) {
        std::cerr << "Failed to create accretion disk particles" << std::endl;
        bh_destroy_particle_system(m_physicsContext, particleSystem);
        return;
    }
    
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
        
        // Update the simulation
        BHErrorCode error = bh_update_particles(m_physicsContext, particleSystem);
        
        if (error != BH_SUCCESS) {
            std::cerr << "Error updating particles" << std::endl;
            break;
        }
        
        // Update simulation time
        simulationTime += scaledTimeStep;
        
        // Get particle data for rendering
        int count = numParticles;
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
            
            // Swap buffers
            swapRenderBuffers();
            
            // Signal that data is ready
            m_dataReady = true;
            m_dataReadyCV.notify_one();
        }
        
        // Sleep to avoid maxing out CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
    
    if (lastMass != m_uiState.mass || 
        lastSpin != m_uiState.spin) {
        
        // Update black hole parameters
        bh_configure_black_hole(
            m_physicsContext,
            m_uiState.mass,
            m_uiState.spin,
            0.0  // No charge
        );
        
        lastMass = m_uiState.mass;
        lastSpin = m_uiState.spin;
    }
    
    if (lastAccretionRate != m_uiState.accretionRate) {
        // Update accretion disk parameters
        bh_configure_accretion_disk(
            m_physicsContext,
            6.0,  // Inner radius
            20.0, // Outer radius
            1.0,  // Temperature scale
            m_uiState.accretionRate // Density scale
        );
        
        lastAccretionRate = m_uiState.accretionRate;
    }
}

void Renderer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    // Update viewport when window is resized
    glViewport(0, 0, width, height);
    
    // Update renderer instance
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (renderer) {
        renderer->m_width = width;
        renderer->m_height = height;
    }
} 