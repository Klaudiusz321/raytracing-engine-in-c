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

// Fragment shader for black hole simulation with proper ray tracing physics
const char* blackHoleFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

// Physics parameters from the backend
uniform float mass;
uniform float spin;
uniform vec3 cameraPos;
uniform int integrationSteps;
uniform float maxDistance;
uniform bool enableDopplerEffect;
uniform bool enableGravitationalRedshift;

// Observer parameters
uniform vec3 observerPosition;
uniform vec3 observerLookDir;
uniform vec3 observerUpDir;
uniform float timeDilation;
uniform float shadowRadius;

// Accretion disk parameters
uniform float diskInnerRadius;
uniform float diskOuterRadius;
uniform float diskTemperature;
uniform int showAccretionDisk;

// Constants
const float PI = 3.14159265359;
const float EPSILON = 1e-5;
const int MAX_STEPS = 1000;

// Converts 3D cartesian coordinates to spherical (r, theta, phi)
vec3 cartesianToSpherical(vec3 cartesian) {
    float r = length(cartesian);
    float theta = acos(cartesian.y / r);
    float phi = atan(cartesian.z, cartesian.x);
    return vec3(r, theta, phi);
}

// Converts spherical coordinates (r, theta, phi) to 3D cartesian
vec3 sphericalToCartesian(vec3 spherical) {
    float r = spherical.x;
    float theta = spherical.y;
    float phi = spherical.z;
    return vec3(
        r * sin(theta) * cos(phi),
        r * cos(theta),
        r * sin(theta) * sin(phi)
    );
}

// 4D state vector: [r, theta, phi, t, dr/dlambda, dtheta/dlambda, dphi/dlambda, dt/dlambda]
// where lambda is the affine parameter

// Calculates Schwarzschild Christoffel symbols for the geodesic equations
void schwarzschildChristoffel(float r, float theta, out float[64] gamma) {
    // Initialize all to zero
    for (int i = 0; i < 64; i++) gamma[i] = 0.0;
    
    float rs = 2.0 * mass; // Schwarzschild radius
    
    // Non-zero components of the Christoffel symbols for Schwarzschild metric
    float g00 = 1.0 - rs/r;
    float g11 = -1.0 / g00;
    
    // Γᵗᵣₜ = Γᵗₜᵣ = M/(r(r-2M))
    gamma[0*16 + 1*4 + 0] = gamma[0*16 + 0*4 + 1] = mass / (r * (r - rs));
    
    // Γʳₜₜ = M(r-2M)/r³
    gamma[1*16 + 0*4 + 0] = mass * (r - rs) / (r*r*r);
    
    // Γʳᵣᵣ = -M/(r(r-2M))
    gamma[1*16 + 1*4 + 1] = -mass / (r * (r - rs));
    
    // Γʳₜₕₜₕ = -(r-2M)
    gamma[1*16 + 2*4 + 2] = -(r - rs);
    
    // Γʳₚₕᵢₚₕᵢ = -(r-2M)sin²θ
    gamma[1*16 + 3*4 + 3] = -(r - rs) * sin(theta) * sin(theta);
    
    // Γᶿᵣᶿ = Γᶿᶿᵣ = 1/r
    gamma[2*16 + 1*4 + 2] = gamma[2*16 + 2*4 + 1] = 1.0/r;
    
    // Γᶿₚₕᵢₚₕᵢ = -sin(θ)cos(θ)
    gamma[2*16 + 3*4 + 3] = -sin(theta) * cos(theta);
    
    // Γᵠᵣᵠ = Γᵠᵠᵣ = 1/r
    gamma[3*16 + 1*4 + 3] = gamma[3*16 + 3*4 + 1] = 1.0/r;
    
    // Γᵠₜₕₑₜₐₚₕᵢ = Γᵠₚₕᵢₜₕₑₜₐ = cot(θ)
    gamma[3*16 + 2*4 + 3] = gamma[3*16 + 3*4 + 2] = 1.0 / tan(theta);
}

// Calculate derivatives for 4D geodesic equations - Schwarzschild
void geodesicDerivatives(vec4 pos, vec4 vel, out vec4 acc) {
    float r = pos.x;
    float theta = pos.y;
    float phi = pos.z;
    
    float dr_dl = vel.x;
    float dtheta_dl = vel.y;
    float dphi_dl = vel.z;
    float dt_dl = vel.w;
    
    // Calculate Christoffel symbols
    float gamma[64];
    schwarzschildChristoffel(r, theta, gamma);
    
    // Geodesic equations for Schwarzschild metric (using Einstein summation convention)
    // d²x^mu/dl² + Γ^mu_αβ * dx^α/dl * dx^β/dl = 0
    
    // Initialize accelerations
    acc = vec4(0.0);
    
    // Calculate acceleration components using Christoffel symbols
    // For each component mu, sum over all alpha and beta
    for (int mu = 0; mu < 4; mu++) {
        for (int alpha = 0; alpha < 4; alpha++) {
            for (int beta = 0; beta < 4; beta++) {
                int idx = mu*16 + alpha*4 + beta;
                float vel_alpha = alpha == 0 ? dt_dl : (alpha == 1 ? dr_dl : (alpha == 2 ? dtheta_dl : dphi_dl));
                float vel_beta = beta == 0 ? dt_dl : (beta == 1 ? dr_dl : (beta == 2 ? dtheta_dl : dphi_dl));
                
                acc[mu] -= gamma[idx] * vel_alpha * vel_beta;
            }
        }
    }
}

// Kerr metric functions for Boyer-Lindquist coordinates
float kerrDelta(float r) {
    float a = spin * mass;
    return r*r - 2.0*mass*r + a*a;
}

float kerrSigma(float r, float theta) {
    float a = spin * mass;
    return r*r + a*a*cos(theta)*cos(theta);
}

// Calculate derivatives for 4D geodesic equations - Kerr metric
void kerrGeodesicDerivatives(vec4 pos, vec4 vel, out vec4 acc) {
    float r = pos.x;
    float theta = pos.y;
    float phi = pos.z;
    float t = pos.w;
    
    float dr_dl = vel.x;
    float dtheta_dl = vel.y;
    float dphi_dl = vel.z;
    float dt_dl = vel.w;
    
    float a = spin * mass;
    float sigma = kerrSigma(r, theta);
    float delta = kerrDelta(r);
    
    // Initialize accelerations
    acc = vec4(0.0);
    
    // r component acceleration
    acc.x = -2.0*r*dr_dl*dr_dl/sigma
           + sin(theta)*sin(theta)*a*a*sin(2.0*theta)*dtheta_dl*dtheta_dl/(2.0*sigma)
           + r*(r*r + a*a)*sin(2.0*theta)*dphi_dl*dphi_dl/(2.0*sigma)
           - 2.0*mass*r*r*dt_dl*dt_dl/pow(sigma, 3.0)
           + a*a*sin(theta)*sin(theta)*r*dphi_dl*dphi_dl/sigma
           - 2.0*a*r*dt_dl*dphi_dl/pow(sigma, 2.0)
           + delta*sigma*dr_dl*dr_dl/(r*r);
    
    // theta component acceleration
    acc.y = -sin(2.0*theta)*a*a*dr_dl*dr_dl/(2.0*sigma)
           - 2.0*r*dr_dl*dtheta_dl/sigma
           + sin(2.0*theta)*dt_dl*dt_dl*a*a/(2.0*pow(sigma, 2.0))
           - (r*r + a*a)*sin(2.0*theta)*dphi_dl*dphi_dl/2.0
           - a*a*sin(2.0*theta)*cos(2.0*theta)*dphi_dl*dphi_dl/2.0
           + a*sin(2.0*theta)*dt_dl*dphi_dl/sigma;
    
    // phi component acceleration
    acc.z = -2.0*dr_dl*dphi_dl/r
           - 2.0*dtheta_dl*dphi_dl/tan(theta)
           + 2.0*a*r*dr_dl*dt_dl/(delta*sigma)
           - 2.0*a*sin(theta)*cos(theta)*dtheta_dl*dt_dl/sigma;
    
    // t component acceleration
    acc.w = -2.0*mass*dr_dl*dt_dl/(sigma*delta)
           + a*sin(theta)*sin(theta)*(r*r + a*a)*dr_dl*dphi_dl/(sigma*delta)
           + 2.0*a*r*sin(theta)*sin(theta)*dtheta_dl*dphi_dl/sigma;
}

// Adaptive step RK4 integration of the geodesic equations
vec3 traceGeodesic(vec3 rayOrigin, vec3 rayDir) {
    // Initial position in spherical coordinates (r, theta, phi)
    vec3 pos_spherical = cartesianToSpherical(rayOrigin);
    float r = pos_spherical.x;
    float theta = pos_spherical.y;
    float phi = pos_spherical.z;
    
    // Initial 4-position (r, theta, phi, t)
    vec4 pos = vec4(r, theta, phi, 0.0);
    
    // Convert direction to spherical coordinate derivatives
    vec3 dir_cartesian = normalize(rayDir);
    float dr_dl = dot(dir_cartesian, rayOrigin / r);
    
    // Calculate tangential components
    vec3 e_theta = vec3(
        cos(theta) * cos(phi),
        -sin(theta),
        cos(theta) * sin(phi)
    );
    
    vec3 e_phi = vec3(
        -sin(phi),
        0.0,
        cos(phi)
    );
    
    float dtheta_dl = dot(dir_cartesian, e_theta) / r;
    float dphi_dl = dot(dir_cartesian, e_phi) / (r * sin(theta));
    
    // For null geodesics (light), we can normalize dt_dl
    float dt_dl = 1.0;
    
    // Initial 4-velocity
    vec4 vel = vec4(dr_dl, dtheta_dl, dphi_dl, dt_dl);
    
    // Integration step size
    float dl = maxDistance / float(integrationSteps);
    
    // RK4 integration of geodesic equations
    for (int i = 0; i < integrationSteps; i++) {
        // Check for event horizon
        if (r <= shadowRadius) {
            return vec3(0.0, 0.0, 0.0); // Black hole
        }
        
        // Check for accretion disk intersection
        vec3 pos_cartesian = sphericalToCartesian(vec3(r, theta, phi));
        float disk_height = 0.1;
        
        if (showAccretionDisk == 1 && 
            abs(pos_cartesian.y) < disk_height && 
            r > diskInnerRadius && r < diskOuterRadius) {
            
            // Calculate disk temperature and color
            float temp = diskTemperature * pow(diskInnerRadius / r, 1.5);
            vec3 diskColor = vec3(1.0, 0.7 * temp, 0.4 * temp * temp);
            
            // Apply Doppler effect if enabled
            if (enableDopplerEffect) {
                // Calculate orbital velocity at this radius
                float v_orbit = sqrt(mass / r);
                
                // Simplified Doppler calculation based on orbital position
                float cos_angle = pos_cartesian.x / sqrt(pos_cartesian.x*pos_cartesian.x + pos_cartesian.z*pos_cartesian.z);
                float doppler = 1.0 + 0.5 * v_orbit * cos_angle;
                
                // Adjust color based on Doppler shift
                diskColor.r *= doppler;
                diskColor.gb *= (1.0 / doppler);
            }
            
            // Apply gravitational redshift if enabled
            if (enableGravitationalRedshift) {
                float redshift = sqrt(1.0 - shadowRadius / r);
                diskColor *= redshift;
            }
            
            return diskColor;
        }
        
        // RK4 integration step
        vec4 k1_pos, k1_vel, k2_pos, k2_vel, k3_pos, k3_vel, k4_pos, k4_vel;
        
        // Use appropriate geodesic equations based on spin
        if (abs(spin) < EPSILON) {
            // Schwarzschild metric
            k1_pos = vel * dl;
            geodesicDerivatives(pos, vel, k1_vel);
            
            k2_pos = (vel + 0.5 * k1_vel * dl) * dl;
            geodesicDerivatives(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel * dl, k2_vel);
            
            k3_pos = (vel + 0.5 * k2_vel * dl) * dl;
            geodesicDerivatives(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel * dl, k3_vel);
            
            k4_pos = (vel + k3_vel * dl) * dl;
            geodesicDerivatives(pos + k3_pos, vel + k3_vel * dl, k4_vel);
        } else {
            // Kerr metric
            k1_pos = vel * dl;
            kerrGeodesicDerivatives(pos, vel, k1_vel);
            
            k2_pos = (vel + 0.5 * k1_vel * dl) * dl;
            kerrGeodesicDerivatives(pos + 0.5 * k1_pos, vel + 0.5 * k1_vel * dl, k2_vel);
            
            k3_pos = (vel + 0.5 * k2_vel * dl) * dl;
            kerrGeodesicDerivatives(pos + 0.5 * k2_pos, vel + 0.5 * k2_vel * dl, k3_vel);
            
            k4_pos = (vel + k3_vel * dl) * dl;
            kerrGeodesicDerivatives(pos + k3_pos, vel + k3_vel * dl, k4_vel);
        }
        
        // Update position and velocity
        pos += (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos) / 6.0;
        vel += (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel) / 6.0;
        
        // Update r, theta, phi for convenience
        r = pos.x;
        theta = pos.y;
        phi = pos.z;
        
        // Adjust theta to avoid singularity
        theta = mod(theta + PI, 2.0 * PI) - PI;
        if (theta < 0.0) theta += PI;
        pos.y = theta;
    }
    
    // Ray escaped to infinity - return background color
    // In a full implementation, this would sample from a starfield or skybox
    return vec3(0.1, 0.2, 0.5);
}

void main() {
    // Convert from texture coordinates to screen space (-1 to 1)
    vec2 uv = TexCoord * 2.0 - 1.0;
    
    // Calculate ray direction from camera parameters
    vec3 rayDir = normalize(observerLookDir + uv.x * cross(observerLookDir, observerUpDir) + uv.y * observerUpDir);
    
    // Trace geodesic through curved spacetime
    vec3 color = traceGeodesic(observerPosition, rayDir);
    
    // Output final color
    FragColor = vec4(color, 1.0);
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
    
    // Initialize GLAD - use the correct function
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }
    
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
    
    // Send basic parameters to shader
    m_blackHoleShader->setFloat("mass", m_uiState.mass);
    m_blackHoleShader->setFloat("spin", m_uiState.spin);
    
    // Send camera position
    m_blackHoleShader->setVec3("cameraPos", 
                              m_camera->position.x, 
                              m_camera->position.y, 
                              m_camera->position.z);
    
    // Integration parameters
    m_blackHoleShader->setInt("integrationSteps", m_uiState.integrationSteps);
    m_blackHoleShader->setFloat("maxDistance", 100.0f);
    
    // Calculate camera basis vectors without GLM
    Vector3D camDir = {
        m_camera->target.x - m_camera->position.x,
        m_camera->target.y - m_camera->position.y,
        m_camera->target.z - m_camera->position.z
    };
    
    // Normalize direction
    float dirLength = sqrt(camDir.x * camDir.x + camDir.y * camDir.y + camDir.z * camDir.z);
    camDir.x /= dirLength;
    camDir.y /= dirLength;
    camDir.z /= dirLength;
    
    // Calculate right with cross product (assuming up is (0,1,0))
    Vector3D camRight = {
        camDir.z,
        0.0f,
        -camDir.x
    };
    
    // Normalize right
    float rightLength = sqrt(camRight.x * camRight.x + camRight.y * camRight.y + camRight.z * camRight.z);
    camRight.x /= rightLength;
    camRight.y /= rightLength;
    camRight.z /= rightLength;
    
    // Calculate up with cross product
    Vector3D camUp = {
        camRight.y * camDir.z - camRight.z * camDir.y,
        camRight.z * camDir.x - camRight.x * camDir.z,
        camRight.x * camDir.y - camRight.y * camDir.x
    };
    
    // Set observer position and direction
    m_blackHoleShader->setVec3("observerPosition", 
                             m_camera->position.x, 
                             m_camera->position.y, 
                             m_camera->position.z);
    m_blackHoleShader->setVec3("observerLookDir", camDir.x, camDir.y, camDir.z);
    m_blackHoleShader->setVec3("observerUpDir", camUp.x, camUp.y, camUp.z);
    
    // Get shadow radius directly from mass (known formula for Schwarzschild black hole)
    float shadowRadius = 2.0f * m_uiState.mass; // Exact Schwarzschild radius
    
    // Set physics parameters
    m_blackHoleShader->setFloat("timeDilation", 1.0f); // Simplified time dilation
    m_blackHoleShader->setFloat("shadowRadius", shadowRadius);
    
    // Set accretion disk parameters
    float diskInnerRadius = 6.0f * m_uiState.mass; // ISCO for non-rotating BH
    float diskOuterRadius = 20.0f * m_uiState.mass;
    m_blackHoleShader->setFloat("diskInnerRadius", diskInnerRadius);
    m_blackHoleShader->setFloat("diskOuterRadius", diskOuterRadius);
    m_blackHoleShader->setFloat("diskTemperature", 1.0f);
    
    // Set boolean uniforms using integers
    m_blackHoleShader->setInt("showAccretionDisk", m_uiState.showAccretionDisk ? 1 : 0);
    m_blackHoleShader->setInt("enableDopplerEffect", m_uiState.enableDopplerEffect ? 1 : 0);
    m_blackHoleShader->setInt("enableGravitationalRedshift", m_uiState.enableGravitationalRedshift ? 1 : 0);
    
    // Draw the quad
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    // Render accretion disk particles if enabled
    if (m_uiState.showAccretionDisk) {
        renderAccretionDiskParticles();
    }
    
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
    // Configure simulation using existing API
    BHErrorCode error = bh_configure_simulation(
        m_physicsContext,
        0.01,  // Time step
        100.0, // Max ray distance
        m_uiState.integrationSteps, // Integration steps
        1e-6   // Tolerance
    );
    
    if (error != BH_SUCCESS) {
        std::cerr << "Failed to configure simulation" << std::endl;
        return;
    }
    
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
    // Skip if particles shouldn't be shown or no data is ready
    if (!m_uiState.showAccretionDisk || !m_dataReady)
        return;

    // Lock the render data
    std::lock_guard<std::mutex> lock(m_renderDataMutex);
    
    // Check if we have any particles
    if (m_renderDataFront->particlePositions.empty())
        return;
    
    // Create simple point shader for particles if not already created
    static const char* particleVertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 projection;
    uniform mat4 view;
    
    void main() {
        gl_Position = projection * view * vec4(aPos, 1.0);
        gl_PointSize = 2.0;
    }
    )";
    
    static const char* particleFragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    
    uniform vec3 particleColor;
    
    void main() {
        FragColor = vec4(particleColor, 1.0);
    }
    )";
    
    // Create shader program for particles
    static GLuint particleShader = 0;
    if (particleShader == 0) {
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
            return;
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
            return;
        }
        
        // Link shaders
        particleShader = glCreateProgram();
        glAttachShader(particleShader, vertexShader);
        glAttachShader(particleShader, fragmentShader);
        glLinkProgram(particleShader);
        
        // Check for errors
        glGetProgramiv(particleShader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(particleShader, 512, NULL, infoLog);
            std::cerr << "Particle shader program linking failed: " << infoLog << std::endl;
            return;
        }
        
        // Clean up
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
    
    // Setup particle positions in a VBO
    GLuint particleVBO, particleVAO;
    glGenBuffers(1, &particleVBO);
    glGenVertexArrays(1, &particleVAO);
    
    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    
    // Copy particle positions to GPU
    std::vector<float> particleData;
    for (const auto& pos : m_renderDataFront->particlePositions) {
        particleData.push_back(pos.x);
        particleData.push_back(pos.y);
        particleData.push_back(pos.z);
    }
    
    glBufferData(GL_ARRAY_BUFFER, particleData.size() * sizeof(float), 
                particleData.data(), GL_STATIC_DRAW);
    
    // Set vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Use particle shader
    glUseProgram(particleShader);
    
    // Create simple matrices manually without GLM
    // Create a simple perspective projection matrix
    float aspect = (float)m_width / (float)m_height;
    float fov = 45.0f * 3.14159f / 180.0f; // Convert to radians
    float near = 0.1f;
    float far = 100.0f;
    float tanHalfFov = tan(fov / 2.0f);
    
    // Simple perspective matrix (column-major)
    float projMatrix[16] = {
        1.0f / (aspect * tanHalfFov), 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / tanHalfFov, 0.0f, 0.0f,
        0.0f, 0.0f, -(far + near) / (far - near), -1.0f,
        0.0f, 0.0f, -(2.0f * far * near) / (far - near), 0.0f
    };
    
    // Create a simple lookAt view matrix from camera data
    Vector3D forward = {
        m_camera->target.x - m_camera->position.x,
        m_camera->target.y - m_camera->position.y,
        m_camera->target.z - m_camera->position.z
    };
    
    // Normalize forward
    float forwardLength = sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    forward.x /= forwardLength;
    forward.y /= forwardLength;
    forward.z /= forwardLength;
    
    // Calculate right with cross product (assuming up is (0,1,0))
    Vector3D right = {
        forward.z,
        0.0f,
        -forward.x
    };
    
    // Normalize right
    float rightLength = sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
    right.x /= rightLength;
    right.y /= rightLength;
    right.z /= rightLength;
    
    // Calculate up with cross product
    Vector3D up = {
        right.y * forward.z - right.z * forward.y,
        right.z * forward.x - right.x * forward.z,
        right.x * forward.y - right.y * forward.x
    };
    
    // Create view matrix (column-major)
    float viewMatrix[16] = {
        static_cast<float>(right.x), static_cast<float>(up.x), static_cast<float>(-forward.x), 0.0f,
        static_cast<float>(right.y), static_cast<float>(up.y), static_cast<float>(-forward.y), 0.0f,
        static_cast<float>(right.z), static_cast<float>(up.z), static_cast<float>(-forward.z), 0.0f,
        -static_cast<float>(right.x * m_camera->position.x + right.y * m_camera->position.y + right.z * m_camera->position.z),
        -static_cast<float>(up.x * m_camera->position.x + up.y * m_camera->position.y + up.z * m_camera->position.z),
        static_cast<float>(forward.x * m_camera->position.x + forward.y * m_camera->position.y + forward.z * m_camera->position.z),
        1.0f
};
    
    // Set uniform values
    GLint projLoc = glGetUniformLocation(particleShader, "projection");
    GLint viewLoc = glGetUniformLocation(particleShader, "view");
    GLint colorLoc = glGetUniformLocation(particleShader, "particleColor");
    
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projMatrix);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, viewMatrix);
    glUniform3f(colorLoc, 1.0f, 0.5f, 0.2f); // Orange color for particles
    
    // Set point size and enable blending
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    
    // Draw particles
    glDrawArrays(GL_POINTS, 0, m_renderDataFront->particlePositions.size());
    
    // Restore state
    glDisable(GL_BLEND);
    glDisable(GL_PROGRAM_POINT_SIZE);
    
    // Cleanup
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &particleVAO);
    glDeleteBuffers(1, &particleVBO);
} 