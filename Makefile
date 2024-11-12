CC=g++
CFLAGS=-g -std=c++11
CFLAGS_FS=$(CFLAGS)  # Additional flags for falsesharing, if any
CFLAGS_MDNN=$(CFLAGS) -fopenmp  # Flags for matmul_dnn_nobias
CFLAGS_MDNN+=-std=c++17
# Linker flags
LDFLAGS_FS=-lpthread
LDFLAGS_MDNN=-lpthread -ldnnl

# Automatically determine the number of threads
maxThread := $(shell grep -c ^processor /proc/cpuinfo)
EXTRAFLAGS := -D Thread=$(maxThread)

# Source and Object files for falsesharing
SRC_FS=falsesharing.cpp
OBJ_FS=$(SRC_FS:.cpp=.o)

# Source and Object files for matmul_dnn_nobias
#SRC_MDNN=matmul_nobias.cpp
SRC_MDNN=matmul_argparse.cpp
OBJ_MDNN=$(SRC_MDNN:.cpp=.o)

SRC_BF16=matmul_bf16.cpp
OBJ_BF16=$(SRC_BF16:.cpp=.o)

SRC_SRL=matmul_sereilbf16.cpp
OBJ_SRL=$(SRC_SRL:.cpp=.o)



# Default target
all: falsesharing matmul_dnn_nobias matmul_args matmul_bf16 matmul_sereilbf16

# Compile the object file for falsesharing
$(OBJ_FS): $(SRC_FS)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS_FS) $(EXTRAFLAGS) -c -o $@ $<

# Compile the object file for matmul_dnn_nobias
$(OBJ_MDNN): $(SRC_MDNN)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS_MDNN) -c -o $@ $<
$(OBJ_BF16): $(SRC_BF16)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS_MDNN) -c -o $@ $<

$(OBJ_SRL): $(SRC_SRL)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS_MDNN) -c -o $@ $<

# Link the final executable for falsesharing
falsesharing: $(OBJ_FS)
	@echo "Linking $@..."
	$(CC) $(CFLAGS_FS) $(EXTRAFLAGS) -o $@ $^ $(LDFLAGS_FS)

# Target for matmul_dnn_nobias
matmul_dnn_nobias: $(OBJ_MDNN)
	@echo "Linking $@..."
	$(CC) $(CFLAGS_MDNN) -o $@ $^ $(LDFLAGS_MDNN)

matmul_args: $(OBJ_MDNN)
	@echo "Linking $@..."
	$(CC) $(CFLAGS_MDNN) -o $@ $^ $(LDFLAGS_MDNN)


matmul_bf16: $(OBJ_BF16)
	@echo "Linking $@..."
	$(CC) $(CFLAGS_MDNN) -o $@ $^ $(LDFLAGS_MDNN)

matmul_sereilbf16: $(OBJ_SRL)
	@echo "Linking $@..."
	$(CC) $(CFLAGS_MDNN) -o $@ $^ $(LDFLAGS_MDNN)
# Clean up
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -f *.o falsesharing matmul_dnn_nobias matmul_args matmul_bf16  matmul_sereilbf16

