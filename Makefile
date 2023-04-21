# Compiler options
CC = nvcc
CFLAGS = -std=c++11 -O3 -arch=sm_30

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin
IN_DIR = input_files
OUT_DIR = output_files

# Files
EXEC = $(BIN_DIR)/batch_processing
SRCS = $(wildcard $(SRC_DIR)/*.cu)
INCS = $(wildcard $(INC_DIR)/*.h)
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
INPUT_FILES = $(wildcard $(IN_DIR)/*.png)
OUTPUT_FILES = $(patsubst $(IN_DIR)/%.png,$(OUT_DIR)/%.png,$(INPUT_FILES))

# Targets
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(INCS)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(EXEC) $(OBJS) $(OUTPUT_FILES)

.PHONY: run
run: $(EXEC)
	./$(EXEC) input_files/jobs.txt

$(OUTPUT_FILES): $(OUT_DIR)/%.png: $(IN_DIR)/%.png $(EXEC)
	./$(EXEC) $< $(basename $(notdir $@)).txt $@
