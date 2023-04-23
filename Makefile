# Compiler options
NVCC := nvcc
NVCCFLAGS := -arch=sm_61
INCLUDE := -I./include

# Source files
SRCS := $(wildcard src/*.cu)
OBJS := $(patsubst %.cu,%.o,$(SRCS))

# Output directories
OUTDIR := output
OBJDIR := obj

# Targets
all: $(OBJS)
    $(NVCC) $(NVCCFLAGS) $(INCLUDE) $(OBJS) -o $(OUTDIR)/batch_launcher

%.o: %.cu
    $(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $(OBJDIR)/$@

clean:
    rm -rf $(OUTDIR)/*
    rm -rf $(OBJDIR)/*

.PHONY: all clean
