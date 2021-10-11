#pragma once

#include <math.h>

// Include Kokkos Headers
#include<Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>


// ===================================================
// ===================================================
struct Boid
{
  KOKKOS_FUNCTION
  Boid() :
    position{0,0}
  {}

  double position[2];
};

//! type alias for creating image
using PngData = Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::OpenMP>;

// ===================================================
// ===================================================
struct Dance
{
  using Flock  = Kokkos::View<Boid*, Kokkos::DefaultExecutionSpace>;
  using VecInt = Kokkos::View<int*, Kokkos::DefaultExecutionSpace>;

  Dance(int nBoids)
    : nBoids(nBoids),
      flock("flock",nBoids),
      flock_new("flock_new",nBoids),
      friends("friends",nBoids),
      ennemies("ennemies",nBoids),
      flock_host()
  {

    flock_host = Kokkos::create_mirror(flock);

  }

  //! number of boids
  int    nBoids;

  //! set (flock) of boids
  Flock  flock;

  //! temp array of boids used to compute postion update (one time step)
  Flock  flock_new;

  //! friend index
  VecInt friends;

  // enemy index
  VecInt ennemies;

  //! mirror of flock data on host (for image rendering)
  Flock::HostMirror flock_host;

}; // struct Dance


// ===================================================
// ===================================================
class MyRandomPool
{

public:

  // define an alias to the random generator pool
  using RGPool_t = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

  // type for hold the random generator state
  using rnd_t = RGPool_t::generator_type;

  // which execution space ? OpenMP, Cuda, ...
  using device_t = RGPool_t::device_type;

  MyRandomPool(uint64_t seed)
    : pool(seed)
  {

  }

  RGPool_t pool;

}; // MyRandomPool

// ===================================================
// ===================================================
void initPositions(Dance& dance, MyRandomPool::RGPool_t& rand_pool);

// ===================================================
// ===================================================
void shuffleFriendsAndEnnemies(Dance& dance, MyRandomPool::RGPool_t& rand_pool);

KOKKOS_INLINE_FUNCTION
double SQR(const double& tmp) {return tmp*tmp;}

KOKKOS_INLINE_FUNCTION
void compute_direction(const Boid& b1, const Boid& b2,  Boid& dir)
{
  double norm = sqrt( (b2.position[0]-b1.position[0])*(b2.position[0]-b1.position[0]) +
                      (b2.position[1]-b1.position[1])*(b2.position[1]-b1.position[1]) );
  if (norm < 1e-6)
  {
    dir.position[0] = 0;
    dir.position[1] = 0;
  }
  else
  {
    dir.position[0] = (b2.position[0]-b1.position[0])/norm;
    dir.position[1] = (b2.position[1]-b1.position[1])/norm;
  }
}

// ===================================================
// ===================================================
void updatePositions(Dance& dance);

// ===================================================
// ===================================================
void renderPositions(PngData data, Dance& dance);

// ===================================================
// ===================================================
void savePositions(std::string prefix, int time, PngData data);
