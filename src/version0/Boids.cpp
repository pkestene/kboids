#include "Boids.h"

#include "io/lodepng.h"

#include <chrono>
#include <iostream>
#include <sstream>

// ===================================================
// ===================================================
void initPositions(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool)
{

  Kokkos::fill_random(boidsData.x, rand_pool, -1.0, 1.0);
  Kokkos::fill_random(boidsData.y, rand_pool, -1.0, 1.0);

  // using rnd_t = MyRandomPool::rnd_t;

  // Kokkos::parallel_for(boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  // {
  //   rnd_t rand_gen = rand_pool.get_state();

  //   // birds positions
  //   boidsData.x(index) = Kokkos::rand<rnd_t,double>::draw(rand_gen, -1.0, 1.0);
  //   boidsData.y(index) = Kokkos::rand<rnd_t,double>::draw(rand_gen, -1.0, 1.0);

  //   // free random gen state, so that it can used by other threads later.
  //   rand_pool.free_state(rand_gen);

  // });

} // BoidsData::initPositions

// ===================================================
// ===================================================
void shuffleFriendsAndEnnemies(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool, float rate)
{

  // rate should be in range [0,1]
  rate = (rate < 0) ? 0 : rate;
  rate = (rate > 1) ? 1 : rate;

  using rnd_t = MyRandomPool::rnd_t;

  Kokkos::parallel_for("shuffleFriendsAndEnnemies",boidsData.nBoids,
    KOKKOS_LAMBDA(const int& index)
    {
      rnd_t rand_gen = rand_pool.get_state();

      float r = Kokkos::rand<rnd_t,float>::draw(rand_gen, 0, 1);

      // shuffle friends and ennemies
      if (r < rate)
      {
        boidsData.friends(index) = Kokkos::rand<rnd_t,int>::draw(rand_gen, boidsData.nBoids);
        boidsData.ennemies(index) = Kokkos::rand<rnd_t,int>::draw(rand_gen, boidsData.nBoids);
      }

      // free random gen state, so that it can used by other threads later.
      rand_pool.free_state(rand_gen);

    });

} // BoidsData::shuffleFriendsAndEnnemies

// ===================================================
// ===================================================
void updatePositions(BoidsData& boidsData)
{

  Kokkos::parallel_for("updatePositions", boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    auto x = boidsData.x(index);
    auto y = boidsData.y(index);

    auto index_friend = boidsData.friends(index);
    auto index_ennemy = boidsData.ennemies(index);

    // rule #1, move towards box center
    double dx = -0.01 * x;
    double dy = -0.01 * y;

    float dir_x, dir_y;

    // rule #2, move towards friend
    compute_direction(x,y,
                      boidsData.x(index_friend),
                      boidsData.y(index_friend),
                      dir_x, dir_y);
    dx += 0.05 * dir_x;
    dy += 0.05 * dir_y;

    // rule #3, move away from ennemy
    compute_direction(x,y,
                      boidsData.x(index_ennemy),
                      boidsData.y(index_ennemy),
                      dir_x, dir_y);
    dx -= 0.03 * dir_x;
    dy -= 0.03 * dir_y;

    // update positions
    boidsData.x_new(index) = x + dx;
    boidsData.y_new(index) = y + dy;

  });

  // swap old and new data
  std::swap(boidsData.x, boidsData.x_new);
  std::swap(boidsData.y, boidsData.y_new);

}

// ===================================================
// ===================================================
void copyPositionsForRendering(BoidsData& boidsData)
{

#ifdef FORGE_ENABLED

  Kokkos::parallel_for("copyPositionsForRendering",boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    auto x = boidsData.x(index);
    auto y = boidsData.y(index);

    boidsData.xy(2*index)   = (float)x;
    boidsData.xy(2*index+1) = (float)y;
  });

#endif

}

// ===================================================
// ===================================================
void renderPositions(PngData data, BoidsData& boidsData)
{

  auto scale = (data.extent(0)-1)/2.0;

  // 1. copy back on host boids positions
  Kokkos::deep_copy(boidsData.x_host, boidsData.x);
  Kokkos::deep_copy(boidsData.y_host, boidsData.y);

  // 2. create image
  //auto pol = Kokkos::RangePolicy<>(Kokkos::OpenMP(), 0, boidsData.nBoids);
  //Kokkos::parallel_for(pol, KOKKOS_LAMBDA(const int& index)


  // reset
  for(int i=0; i<data.extent(0); ++i)
    for(int j=0; j<data.extent(1); ++j)
    {
      data(i,j,0)=0;
      data(i,j,1)=0;
      data(i,j,2)=0;
      data(i,j,3)=255;
    }


  for(int index=0; index<boidsData.nBoids; ++index)
  {
    int i = (int) (floor((boidsData.x_host(index)+1)*scale));
    int j = (int) (floor((boidsData.y_host(index)+1)*scale));

    //printf("%d %d %d %f %f\n",index,i,j,boidsData.x_host(index)+1,boidsData.y_host(index)+1);

    if (i>=0 and i<data.extent(0) and
        j>=0 and j<data.extent(1))
    {
      data(i,j,0) = 255;
      data(i,j,1) = 255;
      data(i,j,2) = 255;
      data(i,j,3) = 255;
    }
  }

}

// ===================================================
// ===================================================
void savePositions(std::string prefix, int time, PngData data)
{
  std::ostringstream outNum;
  outNum.width(7);
  outNum.fill('0');
  outNum << time;

  std::string filename = prefix + "_" + outNum.str() + ".png";

  lodepng_encode32_file(filename.c_str(), data.data(), data.extent(0), data.extent(1));

}
