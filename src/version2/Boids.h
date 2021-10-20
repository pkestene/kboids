#pragma once

#include <math.h>
#include <utility>

// Include Kokkos Headers
#include<Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>


#include "Array.h"

// ===================================================
// ===================================================
// struct Boid
// {
//   KOKKOS_FUNCTION
//   Boid() :
//     pos{0,0},
//     delta_pos{0,0},
//     color(0)
//   {}

//   KOKKOS_FUNCTION
//   Boid(int color) :
//     pos{0,0},
//     delta_pos{0,0},
//     color(color)
//   {}

//   //! boid position
//   float pos[2];

//   //! boid displacement
//   float delta_pos[2];

//  //! boid color (= box index), used to sort boids
//   int color;

// };

// ===================================================
// ===================================================
struct BoidsData
{
  //using Flock    = Kokkos::View<Boid*,  Kokkos::DefaultExecutionSpace>;
  using VecInt   = Kokkos::View<int*,   Kokkos::DefaultExecutionSpace>;
  using VecFloat = Kokkos::View<float*, Kokkos::DefaultExecutionSpace>;

  using VecIntAtomic = Kokkos::View<int*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  using VecFloatAtomic = Kokkos::View<float*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;

  static constexpr float XMIN = 0;
  static constexpr float XMAX = 150;
  static constexpr float YMIN = 0;
  static constexpr float YMAX = 150;

  static constexpr int NBOX_X = 10;
  static constexpr int NBOX_Y = 10;

  BoidsData(int nBoids)
    : nBoids(nBoids),
      x("x",nBoids),
      y("y",nBoids),
      dx("dx",nBoids),
      dy("dy",nBoids),
      ennemies("ennemies",nBoids),
      color("color", nBoids),
      tmp("tmp",nBoids),
      boxCount("box count", NBOX_X*NBOX_Y),
      boxIndex("box count integrated", NBOX_X*NBOX_Y),
      box_x("box average x", NBOX_X*NBOX_Y),
      box_y("box average y", NBOX_X*NBOX_Y),
      box_dx("box average dx", NBOX_X*NBOX_Y),
      box_dy("box average dy", NBOX_X*NBOX_Y),
      x_host(),
      y_host()
#ifdef FORGE_ENABLED
      ,xy("xy",2*nBoids)
#endif
  {
    x_host = Kokkos::create_mirror(x);
    y_host = Kokkos::create_mirror(y);
    resetBoxData();
  }

  void resetBoxData()
  {
    Kokkos::deep_copy(boxCount, 0);
    Kokkos::deep_copy(boxIndex, 0);
    Kokkos::deep_copy(box_x, 0.0);
    Kokkos::deep_copy(box_y, 0.0);
    Kokkos::deep_copy(box_dx, 0.0);
    Kokkos::deep_copy(box_dy, 0.0);
  }

  //! number of boids
  int nBoids;

  //! set of boids coordinates
  VecFloat x, y;

  //! displacement (or velocity)
  VecFloat dx, dy;

  //! ennemies index
  VecInt ennemies;

  //! color (used for sorting)
  VecInt color;

  //! temp array of boids used to perform permutation
  VecFloat tmp;

  //! box population
  VecInt boxCount;

  //! integrated box count
  VecInt boxIndex;

  //! box average position
  VecFloat box_x, box_y;

  //! box average velocity
  VecFloat box_dx, box_dy;

  //! mirror of flock data on host (for image rendering only)
  VecFloat::HostMirror x_host, y_host;

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
/**
 * Randomly change ennemies.
 */
void shuffleEnnemies(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool, float rate);

// ===================================================
// ===================================================
void computeBoxData(BoidsData& boidsData);

// ===================================================
// ===================================================
std::pair<float,float> updateAverageVelocity(BoidsData& boidsData);

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
double SQR(const double& v) {return v*v;}

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
    y = (y2-y1)/norm;
  }
}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
float compute_distance(float x1, float y1, float x2, float y2)
{
  float d = sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
  return d; //(d<1e-6) ? 0 : d;
}

// ===================================================
// ===================================================
template<int dir>
KOKKOS_INLINE_FUNCTION
int pos2box(float x)
{

  auto MIN  = dir == 0 ? BoidsData::XMIN   : BoidsData::YMIN;
  auto MAX  = dir == 0 ? BoidsData::XMAX   : BoidsData::YMAX;
  auto NBOX = dir == 0 ? BoidsData::NBOX_X : BoidsData::NBOX_Y;

  int i = (int) std::floor( (x+MIN)/(MAX-MIN)*NBOX );
  if (i<0) i=0;
  if (i>= NBOX) i=NBOX-1;
  return i;
}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
int pos2box(float x, float y)
{
  int i = pos2box<0>(x);
  int j = pos2box<1>(y);

  return i + BoidsData::NBOX_X * j;
}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
void speedLimit(float& dx, float& dy)
{
  const auto speed = sqrt(dx*dx+dy*dy);
  const float speedLimit = 20;
  if (speed > speedLimit)
  {
    dx = (dx / speed) * speedLimit;
    dy = (dy / speed) * speedLimit;
  }
}

// ===================================================
// ===================================================
KOKKOS_INLINE_FUNCTION
void keepInTheBox(float x, float y, float& dx, float& dy)
{

  float margin = 2*(BoidsData::XMAX-BoidsData::XMIN);

  float turnFactor = 1;

  if (x < BoidsData::XMIN + margin) {
    dx += turnFactor;
  }
  if (x > BoidsData::XMAX - margin) {
    dx -= turnFactor;
  }
  if (y < BoidsData::YMIN + margin) {
    dy += turnFactor;
  }
  if (y > BoidsData::YMAX - margin) {
    dy -= turnFactor;
  }

  // float margin = 0.01*(BoidsData::XMAX-BoidsData::XMIN);
  // if (x > BoidsData::XMIN - margin) {
  //   dx = -dx;
  // }
  // if (x < BoidsData::XMAX + margin) {
  //   dx = -dx;
  // }
  // if (y > BoidsData::YMIN - margin) {
  //   dy = -dy;
  // }
  // if (y < BoidsData::YMAX + margin) {
  //   dy = -dy;
  // }

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

