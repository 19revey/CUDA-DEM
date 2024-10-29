# Discrete Element Method (DEM) Simulation with CUDA GPU Acceleration

This code implements a Discrete Element Method simulation that runs on CUDA-enabled NVIDIA GPUs for high performance particle dynamics calculations.

## Requirements
- CUDA-capable NVIDIA GPU
- CUDA Toolkit (tested with CUDA 8.5)
- C++ compiler with CUDA support
- Make build system

## Setup & Configuration
1. Clone this repository
2. Edit the Makefile:
   - Specify your CUDA compiler path (search #TODO in makefile to Set CUDA and g++ path)
   - Adjust CUDA architecture flags if needed

## Building & Running

make clean
make all
./bin/particles

