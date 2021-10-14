#include "Boids.h"

#include "io/lodepng.h"

#include <chrono>
#include <iostream>
#include <sstream>

// ===================================================
// ===================================================
void initPositions(Dance& dance, MyRandomPool::RGPool_t& rand_pool)
{

  using rnd_t = MyRandomPool::rnd_t;

  Kokkos::parallel_for(dance.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    rnd_t rand_gen = rand_pool.get_state();

    // shuffle friends and ennemies
    dance.flock(index).position[0] = Kokkos::rand<rnd_t,double>::draw(rand_gen, -1.0, 1.0);
    dance.flock(index).position[1] = Kokkos::rand<rnd_t,double>::draw(rand_gen, -1.0, 1.0);

    // free random gen state, so that it can used by other threads later.
    rand_pool.free_state(rand_gen);

  });

} // Dance::initPositions

// ===================================================
// ===================================================
void shuffleFriendsAndEnnemies(Dance& dance, MyRandomPool::RGPool_t& rand_pool)
{

  using rnd_t = MyRandomPool::rnd_t;

  Kokkos::parallel_for(dance.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    rnd_t rand_gen = rand_pool.get_state();

    // shuffle friends and ennemies
    dance.friends(index) = Kokkos::rand<rnd_t,int>::draw(rand_gen, dance.nBoids);
    dance.ennemies(index) = Kokkos::rand<rnd_t,int>::draw(rand_gen, dance.nBoids);

    // free random gen state, so that it can used by other threads later.
    rand_pool.free_state(rand_gen);

  });

} // Dance::shuffleFriendsAndEnnemies

// ===================================================
// ===================================================
void updatePositions(Dance& dance)
{

  Kokkos::parallel_for(dance.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    auto x = dance.flock(index).position[0];
    auto y = dance.flock(index).position[1];

    auto index_friend = dance.friends(index);
    auto index_ennemy = dance.ennemies(index);

    // rule #1, move towards box center
    double dx = -0.01 * x;
    double dy = -0.01 * y;

    Boid dir;

    // rule #2, move towards friend
    compute_direction(dance.flock(index),dance.flock(index_friend), dir);
    dx += 0.05 * dir.position[0];
    dy += 0.05 * dir.position[1];

    // rule #3, move away from ennemy
    compute_direction(dance.flock(index),dance.flock(index_ennemy), dir);
    dx -= 0.03 * dir.position[0];
    dy -= 0.03 * dir.position[1];

    // update positions
    dance.flock_new(index).position[0] = x + dx;
    dance.flock_new(index).position[1] = y + dy;

    //printf("%d | %f %f | %f %f\n",index,x,y,x+dx,y+dy);


  });

  // swap flock and flock_new
  {
    Dance::Flock tmp = dance.flock;
    dance.flock      = dance.flock_new;
    dance.flock_new  = tmp;
  }

}

// ===================================================
// ===================================================
void renderPositions(PngData data, Dance& dance)
{

  auto scale = (data.extent(0)-1)/2.0;

  // 1. copy back on host boids positions
  Kokkos::deep_copy(dance.flock_host, dance.flock);

  // 2. create image
  //auto pol = Kokkos::RangePolicy<>(Kokkos::OpenMP(), 0, dance.nBoids);
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


  for(int index=0; index<dance.nBoids; ++index)
  {
    int i = (int) (floor((dance.flock_host(index).position[0]+1)*scale));
    int j = (int) (floor((dance.flock_host(index).position[1]+1)*scale));

    //printf("%d %d %d %f %f\n",index,i,j,dance.flock_host(index).position[0]+1,dance.flock_host(index).position[1]+1);

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
