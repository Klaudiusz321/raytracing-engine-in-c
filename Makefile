CC = gcc
CFLAGS = -Wall -Wextra -g -O2 -I./include -std=c99
LDFLAGS = -lm

# Source files
SRC_DIR = src
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:.c=.o)

# Output binary
BIN_DIR = bin
TARGET = $(BIN_DIR)/blackhole_sim

# Default target
all: directories $(TARGET)

# Create required directories
directories:
	@mkdir -p $(BIN_DIR)

# Link object files to create the executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(TARGET)

# Clean up and rebuild
rebuild: clean all

# Run the test program
run: all
	$(TARGET)

# Create a static library
lib: $(filter-out $(SRC_DIR)/main.o, $(OBJECTS))
	ar rcs $(BIN_DIR)/libblackhole.a $^

# Build WebAssembly version (requires Emscripten)
wasm: $(SOURCES)
	@mkdir -p $(BIN_DIR)/web
	emcc $(CFLAGS) -s WASM=1 -s EXPORTED_FUNCTIONS="['_bh_initialize', '_bh_shutdown', '_bh_configure_black_hole', '_bh_configure_accretion_disk', '_bh_configure_simulation', '_bh_trace_ray', '_bh_trace_rays_batch', '_bh_create_particle_system', '_bh_destroy_particle_system', '_bh_add_test_particle', '_bh_create_accretion_disk_particles', '_bh_generate_hawking_radiation', '_bh_update_particles', '_bh_get_particle_data', '_bh_calculate_time_dilation', '_bh_get_version', '_malloc', '_free']" -s EXPORTED_RUNTIME_METHODS="['ccall', 'cwrap']" -o $(BIN_DIR)/web/blackhole.js $(SOURCES) $(LDFLAGS)

.PHONY: all clean rebuild run lib wasm directories 