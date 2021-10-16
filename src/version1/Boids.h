#pragma once

#include <math.h>

// Include Kokkos Headers
#include<Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>


#include "Array.h"

// ===================================================
// ===================================================
struct Boid
{
  KOKKOS_FUNCTION
  Boid() :
    pos{0,0},
    delta_pos{0,0},
    color(0)
  {}

  KOKKOS_FUNCTION
  Boid(int color) :
    pos{0,0},
    delta_pos{0,0},
    color(color)
  {}

  //! boid position
  float pos[2];

  //! boid displacement
  float delta_pos[2];

  //! boid color (= box index), used to sort boids
  int color;

};

// necessary for custom sort
// neutral element for max/min operation
namespace Kokkos {
template <>
struct reduction_identity<Boid> {

  KOKKOS_FORCEINLINE_FUNCTION static Boid max() {
    return Boid(INT_MAX);
  }
  KOKKOS_FORCEINLINE_FUNCTION static Boid min() {
    return Boid(INT_MIN);
  }
};
} // namespace Kokkos

// ===================================================
// ===================================================
struct BoidsData
{
  using Flock    = Kokkos::View<Boid*,  Kokkos::DefaultExecutionSpace>;
  using VecInt   = Kokkos::View<int*,   Kokkos::DefaultExecutionSpace>;
  using VecFloat = Kokkos::View<float*, Kokkos::DefaultExecutionSpace>;

  static constexpr int NBOX = 10;

  BoidsData(int nBoids)
    : nBoids(nBoids),
      flock("flock",nBoids),
      flock_new("flock_new",nBoids),
      boxCount("box count", NBOX*NBOX),
      boxIndex("box count integrated", NBOX*NBOX),
      flock_host()
#ifdef FORGE_ENABLED
      ,xy("xy",2*nBoids)
#endif
  {
    flock_host = Kokkos::create_mirror(flock);
    resetBoxData();
  }

  void resetBoxData()
  {
    Kokkos::deep_copy(boxCount, 0);
    Kokkos::deep_copy(boxIndex, 0);
  }

  //! number of boids
  int nBoids;

  //! set (flock) of boids
  Flock flock;

  //! temp array of boids used to compute postion update (one time step)
  Flock flock_new;

  //! box population
  VecInt boxCount;

  //! integrated box count
  VecInt boxIndex;

  //! mirror of flock data on host (for image rendering only)
  Flock::HostMirror flock_host;

#ifdef FORGE_ENABLED
  VecFloat xy;
#endif

}; // struct BoidsData

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
void initPositions(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool);

// ===================================================
// ===================================================
void computeBoxData(BoidsData& boidsData);

// ===================================================
// ===================================================
void updateAverageVelocity(BoidsData& boidsData, float& vx, float& vy);

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
double SQR(const double& tmp) {return tmp*tmp;}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
void compute_direction(const float& x1, const float& y1,
                       const float& x2, const float& y2,
                       float& x, float& y)
{
  double norm = sqrt( (x2-x1)*(x2-x1) +
                      (y2-y1)*(y2-y1) );
  if (norm < 1e-6)
  {
    x = 0;
    y = 0;
  }
  else
  {
    x = (x2-x1)/norm;
    y = (y2-y2)/norm;
  }
}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
float compute_distance(float x1, float y1, float x2, float y2)
{
  float d = sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
  return (d<1e-6) ? 0 : d;
}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
int pos2box(float x, int NBOX)
{
  int i = (int) std::round( (x+1)/2*NBOX );
  if (i<0) i=0;
  if (i>= NBOX) i=NBOX-1;
  return i;
}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
int pos2box(float x, float y, int NBOX)
{
  int i = pos2box(x,NBOX);
  int j = pos2box(y,NBOX);

  return i+NBOX*j;
}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
void speedLimit(float& dx, float& dy)
{
  const auto speed = sqrt(dx*dx+dy*dy);
  if (speed > 0)
  {
    dx = (dx/speed) * 0.2;
    dy = (dy/speed) * 0.2;
  }
}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
void keepInTheBox(float x, float y, float& dx, float& dy)
{

  float margin = 0.1;

  float turnFactor = 0.01;

  if (x < -1.0 + margin) {
    dx += turnFactor;
  }
  if (x > 1.0 - margin) {
    dx -= turnFactor;
  }
  if (y < -1.0 + margin) {
    dy += turnFactor;
  }
  if (y > 1.0 - margin) {
    dy -= turnFactor;
  }

}

// ===================================================
// ===================================================
void computeBoxIndex(BoidsData& boidsData);

// ===================================================
// ===================================================
void updatePositions(BoidsData& boidsData);

// ===================================================
// ===================================================
void copyPositionsForRendering(BoidsData& boidsData);

