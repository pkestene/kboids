#include "Boids.h"

#include "io/lodepng.h"

#include <chrono>
#include <iostream>
#include <sstream>

#include <Kokkos_Sort.hpp>

// ===================================================
// ===================================================
void initPositions(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool)
{

  using rnd_t = MyRandomPool::rnd_t;

  Kokkos::parallel_for("initPositions", boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    rnd_t rand_gen = rand_pool.get_state();

    // position
    boidsData.flock(index).pos[0] = Kokkos::rand<rnd_t,double>::draw(rand_gen, -1.0, 1.0);
    boidsData.flock(index).pos[1] = Kokkos::rand<rnd_t,double>::draw(rand_gen, -1.0, 1.0);

    // delta position
    boidsData.flock(index).delta_pos[0] = Kokkos::rand<rnd_t,double>::draw(rand_gen, -0.01, 0.01);
    boidsData.flock(index).delta_pos[1] = Kokkos::rand<rnd_t,double>::draw(rand_gen, -0.01, 0.01);

    // free random gen state, so that it can used by other threads later.
    rand_pool.free_state(rand_gen);

  });

} // BoidsData::initPositions

// ===================================================
// ===================================================
void updateAverageVelocity(BoidsData& boidsData, float& vx, float& vy)
{

  // this is just a reduction
  // we could also use a custom reducer, and place output directly in
  // device memory
  // see https://github.com/kokkos/kokkos/wiki/Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types

  Kokkos::parallel_reduce("updateAverageVelocity x", boidsData.nBoids,
     KOKKOS_LAMBDA(const int index, float& value)
     {
       value += boidsData.flock(index).delta_pos[0];
     }, vx);

  Kokkos::parallel_reduce("updateAverageVelocity y", boidsData.nBoids,
     KOKKOS_LAMBDA(const int index, float& value)
     {
       value += boidsData.flock(index).delta_pos[1];
     }, vy);

  vx /= boidsData.nBoids;
  vy /= boidsData.nBoids;

}

// ===================================================
// ===================================================
void computeBoxData(BoidsData& boidsData)
{

  boidsData.resetBoxData();

  // compute the number of boids per box
  // and compute boids color
  Kokkos::parallel_for("computeBoxCount",
                       boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    auto x = boidsData.flock(index).pos[0];
    auto y = boidsData.flock(index).pos[1];

    int iBox = pos2box(x,y,BoidsData::NBOX);

    Kokkos::atomic_fetch_add( &(boidsData.boxCount(iBox)), 1);

    // update current boid color
    boidsData.flock(index).color = iBox;

  });

  // sort boid's flock per color
  //Kokkos::sort(boidsData.flock, true);

  // compute index to first boids of each color
  // using exclusive scan pattern
  Kokkos::parallel_scan(boidsData.nBoids,
     KOKKOS_LAMBDA(const int iBox,
                   int& update, const bool final)
     {
       const int iTmp = boidsData.boxCount(iBox);

       if (final)
         boidsData.boxIndex(iBox) = update;

       update += iTmp;
     });

}

// ===================================================
// ===================================================
void updatePositions(BoidsData& boidsData)
{

  // prepare data used in rule #2
  float dx_av, dy_av;
  updateAverageVelocity(boidsData, dx_av, dy_av);

  // prepare data used in rule #3
  // i.e. move away from close neighbors
  computeBoxData(boidsData);


  Kokkos::parallel_for(boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    auto x = boidsData.flock(index).pos[0];
    auto y = boidsData.flock(index).pos[1];

    auto delta_x = boidsData.flock(index).delta_pos[0];
    auto delta_y = boidsData.flock(index).delta_pos[1];

    // rule #1, move towards box center
    float dx = -0.01 * x;
    float dy = -0.01 * y;

    // rule #2, update dx,dy by a restoring step toward
    // average velocity
    dx += 0.05 * (delta_x - dx_av);
    dy += 0.05 * (delta_y - dy_av);

    // int i = pos2box(x,BoidsData::NBOX);
    // int j = pos2box(y,BoidsData::NBOX);

    // auto count = boidsData.boxCount(i,j);
    // rule #3, move away from close neighbors
    // float dir_x, dir_y;
    // auto ie = boidsData.flock(index).ennemy;
    // float xe = boidsData.flock(ie).pos[0];
    // float ye = boidsData.flock(ie).pos[1];
    // compute_direction(x, y, xe, ye, dir_x, dir_y);
    // dx -= 0.00 * dir_x;
    // dy -= 0.00 * dir_y;

    // speed limit
    speedLimit(dx,dy);

    // keep in the box
    keepInTheBox(x,y,dx,dy);

    // update positions
    boidsData.flock_new(index).pos[0] = x + dx;
    boidsData.flock_new(index).pos[1] = y + dy;

    // update delta_pos
    boidsData.flock_new(index).delta_pos[0] = dx;
    boidsData.flock_new(index).delta_pos[1] = dy;

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
    auto x = boidsData.flock(index).pos[0];
    auto y = boidsData.flock(index).pos[1];

    boidsData.xy(2*index)   = (float)x;
    boidsData.xy(2*index+1) = (float)y;
  });

#endif

}
