#pragma once

#include <math.h>

// Include Kokkos Headers
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>

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
template<typename ExecutionSpace>
class MyRandomPool
{

public:

  // define an alias to the random generator pool
  using RGPool_t = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;

  // type for hold the random generator state
  using rnd_t = typename RGPool_t::generator_type;

  MyRandomPool(uint64_t seed)
    : pool(seed)
  {

  }

  RGPool_t pool;

}; // MyRandomPool


//! type alias for creating image on host
using PngData = Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::Serial>;

// ===================================================
// ===================================================
template <typename DeviceType>
class BoidsData
{

public:

  using device            = DeviceType;
  using execution_space   = typename DeviceType::execution_space;
  using memory_space      = typename DeviceType::memory_space;

  using myrandom_pool = MyRandomPool<execution_space>;
  using RGPool_t      = typename myrandom_pool::RGPool_t;

  using VecInt   = Kokkos::View<int*,   memory_space>;
  using VecFloat = Kokkos::View<float*, memory_space>;
  using VecFloatMirror = typename VecFloat::HostMirror;

  //! \param[in] nBoids is the number of boids
  //! \param[in] seed is the random number generator seed
  BoidsData(int nBoids, uint64_t seed)
    : m_nBoids(nBoids),
      m_rand_pool(seed),
      m_x("x",nBoids),
      m_y("y",nBoids),
      m_x_new("x_new",nBoids),
      m_y_new("y_new",nBoids),
      m_friends("friends",nBoids),
      m_ennemies("ennemies",nBoids),
      m_x_host(),
      m_y_host()
#ifdef FORGE_ENABLED
      ,m_xy("xy",2*nBoids)
#endif
  {
    m_x_host = Kokkos::create_mirror(m_x);
    m_y_host = Kokkos::create_mirror(m_y);
  }

  /**
   * initialize boids positions.
   */
  void initPositions();

  /**
   * Randomly change friends and ennemies.
   *
   * \param[in] exec_space is an execution instance.
   *
   * Not required here, but in some situation (Kokkos::Cuda) you might need to retrieve a cuda_stream_t
   * from it.
   *
   * \param[in] rate is ratio (in [0,1]) : we randomly chose particles and refresh their friend and ennemy mate index.
   */
  void shuffleFriendsAndEnnemies(execution_space const &exec_space, float rate);

  /**
   * update boids positions (one time step).
   */
  void updatePositions(execution_space const &exec_space);

  //! render positions to PNG image
  void renderPositions(PngData data);

  //! copy data for OpenGL rendering
  void copyPositionsForOpenGLRendering(execution_space const &exec_space);

private:
  //! number of boids
  int m_nBoids;

  //! random number generator
  RGPool_t m_rand_pool;

  //! set of boids coordinates
  VecFloat m_x, m_y;

  //! temp array of boids used to compute postion update (one time step)
  VecFloat m_x_new, m_y_new;

  //! vector of friend index
  VecInt m_friends;

  //! vector of enemy index
  VecInt m_ennemies;

public:
  //! mirror of x,y data on host (for image rendering only)
  VecFloatMirror m_x_host;
  VecFloatMirror m_y_host;

#ifdef FORGE_ENABLED
  VecFloat m_xy;
#endif

}; // struct BoidsData

// ===================================================
// ===================================================
template<typename DeviceType>
void BoidsData<DeviceType>::initPositions()
{

  Kokkos::fill_random(m_x, m_rand_pool, -1.0, 1.0);
  Kokkos::fill_random(m_y, m_rand_pool, -1.0, 1.0);

}

// ===================================================
// ===================================================
template<typename DeviceType>
void BoidsData<DeviceType>::shuffleFriendsAndEnnemies(execution_space const &exec_space, float rate)
{

  // rate should be in range [0,1]
  rate = (rate < 0) ? 0 : rate;
  rate = (rate > 1) ? 1 : rate;

  using rnd_t = typename myrandom_pool::rnd_t;

  auto policy = Kokkos::RangePolicy<execution_space>(exec_space, 0, m_nBoids);

  Kokkos::parallel_for("shuffleFriendsAndEnnemies", policy, KOKKOS_CLASS_LAMBDA(const int& index)
  {
    rnd_t rand_gen = m_rand_pool.get_state();

    float r = Kokkos::rand<rnd_t,float>::draw(rand_gen, 0, 1);

    // shuffle friends and ennemies
    if (r < rate)
    {
      m_friends(index)  = Kokkos::rand<rnd_t,int>::draw(rand_gen, m_nBoids);
      m_ennemies(index) = Kokkos::rand<rnd_t,int>::draw(rand_gen, m_nBoids);
    }

    // free random gen state, so that it can used by other threads later.
    m_rand_pool.free_state(rand_gen);

  });

} // BoidsData<DeviceType>::shuffleFriendsAndEnnemies

// ===================================================
// ===================================================
template<typename DeviceType>
void BoidsData<DeviceType>::updatePositions(execution_space const &exec_space)
{

  auto policy = Kokkos::RangePolicy<execution_space>(exec_space, 0, m_nBoids);

  Kokkos::parallel_for("updatePositions", policy, KOKKOS_CLASS_LAMBDA(const int& index)
  {
    auto x = m_x(index);
    auto y = m_y(index);

    auto index_friend = m_friends(index);
    auto index_ennemy = m_ennemies(index);

    // rule #1, move towards box center
    double dx = -0.01 * x;
    double dy = -0.01 * y;

    float dir_x, dir_y;

    // rule #2, move towards friend
    compute_direction(x,y,
                      m_x(index_friend),
                      m_y(index_friend),
                      dir_x, dir_y);
    dx += 0.05 * dir_x;
    dy += 0.05 * dir_y;

    // rule #3, move away from ennemy
    compute_direction(x,y,
                      m_x(index_ennemy),
                      m_y(index_ennemy),
                      dir_x, dir_y);
    dx -= 0.03 * dir_x;
    dy -= 0.03 * dir_y;

    // update positions
    m_x_new(index) = x + dx;
    m_y_new(index) = y + dy;

  });

  // swap old and new data
  std::swap(m_x, m_x_new);
  std::swap(m_y, m_y_new);

} // BoidsData<DeviceType>::updatePositions

// ===================================================
// ===================================================
template<typename DeviceType>
void BoidsData<DeviceType>::renderPositions(PngData data)
{

  auto scale = (data.extent(0)-1)/2.0;

  // 1. copy back on host boids positions
  Kokkos::deep_copy(m_x_host, m_x);
  Kokkos::deep_copy(m_y_host, m_y);

  // 2. create image
  //auto policy = Kokkos::RangePolicy<>(Kokkos::OpenMP{}, 0, m_nBoids);
  //Kokkos::parallel_for("renderPositions", policy, KOKKOS_CLASS_LAMBDA(const int& index)

  // reset
  for(int i=0; i<data.extent(0); ++i)
    for(int j=0; j<data.extent(1); ++j)
    {
      data(i,j,0)=0;
      data(i,j,1)=0;
      data(i,j,2)=0;
      data(i,j,3)=255;
    }


  for(int index=0; index<m_nBoids; ++index)
  {
    int i = (int) (floor((m_x_host(index)+1)*scale));
    int j = (int) (floor((m_y_host(index)+1)*scale));

    //printf("%d %d %d %f %f\n",index,i,j,m_x_host(index)+1,m_y_host(index)+1);

    if (i>=0 and i<data.extent(0) and
        j>=0 and j<data.extent(1))
    {
      data(i,j,0) = 255;
      data(i,j,1) = 255;
      data(i,j,2) = 255;
      data(i,j,3) = 255;
    }
  }

} // BoidsData<DeviceType>::renderPositions

// ===================================================
// ===================================================
template<typename DeviceType>
void BoidsData<DeviceType>::copyPositionsForOpenGLRendering(execution_space const &exec_space)
{

#ifdef FORGE_ENABLED

  auto policy = Kokkos::RangePolicy<execution_space>(exec_space, 0, m_nBoids);

  Kokkos::parallel_for("copyPositionsForOpenGLRendering", policy, KOKKOS_CLASS_LAMBDA(const int& index)
  {
    auto x = m_x(index);
    auto y = m_y(index);

    m_xy(2*index)   = (float)x;
    m_xy(2*index+1) = (float)y;
  });

#endif

} // BoidsData<DeviceType>::copyPositionsForOpenGLRendering

// ===================================================
// ===================================================
void savePositions(std::string prefix, int time, PngData data);
