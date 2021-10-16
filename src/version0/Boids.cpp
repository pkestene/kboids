#include "Boids.h"

#include "io/lodepng.h"

#include <chrono>
#include <iostream>
#include <sstream>

// ===================================================
// ===================================================
void initPositions(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool)
{

  using rnd_t = MyRandomPool::rnd_t;

  Kokkos::parallel_for(boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    rnd_t rand_gen = rand_pool.get_state();

    // birds positions
    boidsData.flock(index).position[0] = Kokkos::rand<rnd_t,double>::draw(rand_gen, -1.0, 1.0);
    boidsData.flock(index).position[1] = Kokkos::rand<rnd_t,double>::draw(rand_gen, -1.0, 1.0);

    // free random gen state, so that it can used by other threads later.
    rand_pool.free_state(rand_gen);

  });

} // BoidsData::initPositions

// ===================================================
// ===================================================
void shuffleFriendsAndEnnemies(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool, float rate)
{

  // rate should be in range [0,1]
  rate = (rate < 0) ? 0 : rate;
  rate = (rate > 1) ? 1 : rate;

  using rnd_t = MyRandomPool::rnd_t;

  Kokkos::parallel_for(boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
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

  Kokkos::parallel_for(boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    auto x = boidsData.flock(index).position[0];
    auto y = boidsData.flock(index).position[1];

    auto index_friend = boidsData.friends(index);
    auto index_ennemy = boidsData.ennemies(index);

    // rule #1, move towards box center
    double dx = -0.01 * x;
    double dy = -0.01 * y;

    Boid dir;

    // rule #2, move towards friend
    compute_direction(boidsData.flock(index),boidsData.flock(index_friend), dir);
    dx += 0.05 * dir.position[0];
    dy += 0.05 * dir.position[1];

    // rule #3, move away from ennemy
    compute_direction(boidsData.flock(index),boidsData.flock(index_ennemy), dir);
    dx -= 0.03 * dir.position[0];
    dy -= 0.03 * dir.position[1];

    // update positions
    boidsData.flock_new(index).position[0] = x + dx;
    boidsData.flock_new(index).position[1] = y + dy;

    //printf("%d | %f %f | %f %f\n",index,x,y,x+dx,y+dy);


  });

  // swap flock and flock_new
  {
    BoidsData::Flock tmp = boidsData.flock;
    boidsData.flock      = boidsData.flock_new;
    boidsData.flock_new  = tmp;
  }

}

// ===================================================
// ===================================================
void copyPositionsForRendering(BoidsData& boidsData)
{

#ifdef FORGE_ENABLED

  Kokkos::parallel_for(boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    auto x = boidsData.flock(index).position[0];
    auto y = boidsData.flock(index).position[1];

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
  Kokkos::deep_copy(boidsData.flock_host, boidsData.flock);

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
    int i = (int) (floor((boidsData.flock_host(index).position[0]+1)*scale));
    int j = (int) (floor((boidsData.flock_host(index).position[1]+1)*scale));

    //printf("%d %d %d %f %f\n",index,i,j,boidsData.flock_host(index).position[0]+1,boidsData.flock_host(index).position[1]+1);

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
