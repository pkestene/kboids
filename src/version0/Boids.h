#pragma once

#include <math.h>

// Include Kokkos Headers
#include<Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>


// ===================================================
// ===================================================
// struct Boid
// {
//   KOKKOS_FUNCTION
//   Boid() :
//     position{0,0}
//   {}

//   double position[2];
// };

//! type alias for creating image
using PngData = Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::Serial>;

// ===================================================
// ===================================================
struct BoidsData
{
  using VecInt   = Kokkos::View<int*,   Kokkos::DefaultExecutionSpace>;
  using VecFloat = Kokkos::View<float*, Kokkos::DefaultExecutionSpace>;

  BoidsData(int nBoids)
    : nBoids(nBoids),
      x("x",nBoids),
      y("y",nBoids),
      x_new("x_new",nBoids),
      y_new("y_new",nBoids),
      friends("friends",nBoids),
      ennemies("ennemies",nBoids),
      x_host(),
      y_host()
#ifdef FORGE_ENABLED
      ,xy("xy",2*nBoids)
#endif
  {
    x_host = Kokkos::create_mirror(x);
    y_host = Kokkos::create_mirror(y);
  }

  //! number of boids
  int nBoids;

  //! set of boids coordinates
  VecFloat x, y;

  //! temp array of boids used to compute postion update (one time step)
  VecFloat x_new, y_new;

  //! vector of friend index
  VecInt friends;

  //! vector of enemy index
  VecInt ennemies;

  //! mirror of x,y data on host (for image rendering only)
  VecFloat::HostMirror x_host;
  VecFloat::HostMirror y_host;

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
 * Randomly change friends and ennemies.
 */
void shuffleFriendsAndEnnemies(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool, float rate);

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
void updatePositions(BoidsData& boidsData);

// ===================================================
// ===================================================
void renderPositions(PngData data, BoidsData& boidsData);

// ===================================================
// ===================================================
void copyPositionsForRendering(BoidsData& boidsData);

// ===================================================
// ===================================================
void savePositions(std::string prefix, int time, PngData data);
