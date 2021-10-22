# [Boids](https://en.wikipedia.org/wiki/Boids) flight simulation

Just a toy example code whose sole purpose is to help learning [Kokkos/C++ library](https://github.com/kokkos/kokkos).

## What is boids flight ?

It is a simple model for simulating the flocking behaviour of birds.

![boids_512x512](https://github.com/pkestene/kboids/blob/master/boids_flight.gif)

## Software algorithm

In this code, we implement two versions:
- a simplified version : birds moves according to 3 simple rules
  * move towards the box center
  * move towards a _friend_
  * move away from an _ennemy_
  Friends and ennemies are initialized randomly, and then changed once in a while.
- a more elaborated version (**unfinished**) : birds moves according to 3 rules
  * move toward gravity center (barycenter of all other birds)
  * adjust velocity to the avraged velocity of all the birds
  * avoids nearby birds (all the in a small surrounding sphere)

## Software implementation

We revisit the implementation of this simple application using the [Kokkos](https://github.com/kokkos/kokkos) programing model in C++.

- version0 is the most naive implementation, Kokkos device is chosen at compile time
- version1 is a bit more elaborated, we just refactored version0 to allow target backed to be chosen at run time; it also illustrate the use of lambda function inside class
- version 2 (**unfinished**): it illustrates all the parallel patterns available in Kokkos (_parallel_for_, _parallel_reduce_ and _parallel_scan_), atomic access to memory, binning and sorting objects.

Going further (TODO):
- explore SIMD possibilities (through e.g. [kokkos/simd-math](https://github.com/kokkos/simd-math))
- explore the benefit of using a AoSoA (Array-of-Structure-of-Array) data structure and also the linked-cell list algorithms provided by [Cabana](https://github.com/ECP-copa/Cabana)

## Software requirements

All required dependencies are included as a git submodule ([kokkos](https://github.com/kokkos/kokkos), [argh](https://github.com/adishavit/argh))

Optionnal dependencies:
- [likwid](https://github.com/RRZE-HPC/likwid) built with CUDA interface activated
- [Forge](https://github.com/arrayfire/forge) for OpenGL interface (compatible CPU/GPU) from [ArrayFire](https://github.com/arrayfire/arrayfire)
- [Thrust](https://github.com/NVIDIA/thrust) for efficient sort algorithms implementation on Nvidia GPUs

## Get the sources

```shell
git clone git@github.com:pkestene/kboids.git
cd kboids
git submodule update --recursive --init
```

## Build the code

For a CPU/OpenMP build:

```shell
# for Kokkos::OpenMP as default device
mkdir -p _build/openmp
cd _build/openmp
cmake -DKokkos_ENABLE_OPENMP:BOOL=ON ../..
```

You also navigate the cmake build option using `ccmake` and tune some architecture related flags.

For a CGU/CUDA build, you just need to have Nvidia's `nvcc` compiler in your path, cmake will use a compiler wrapper named `nvcc_wrapper` to build the application.

```shell
# for Kokkos::Cuda as default device
mkdir -p _build/cuda
cd _build/cuda
cmake -DKokkos_ENABLE_CUDA:BOOL=ON -DKokkos_ARCH_TURING75:BOOL=ON -DKokkos_ENABLE_CUDA_LAMBDA:BOOL=ON -DCUB_DIR=$CUDA/DIR/include/cub/cmake ../..
```

You can adjust the host compiler by setting environment variable `NVCC_WRAPPER_DEFAULT_COMPILER` (default is GNU g++).

Example with SYCL as device
```shell
# for Kokkos::SYCL as default device
# make sure you have compiler clang++ with sycl capability (https://github.com/intel/llvm)
mkdir -p _build/sycl
cd _build/sycl
cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda" -DKokkos_ENABLE_SYCL=ON ../..
```
