#include "Boids.h"
#include "io/lodepng.h"
#include "utils/sort-utils.h"

#include <chrono>
#include <iostream>
#include <sstream>

// ===================================================
// ===================================================
void initPositions(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool)
{

  Kokkos::fill_random(boidsData.x,  rand_pool, BoidsData::XMIN, BoidsData::XMAX);
  Kokkos::fill_random(boidsData.y,  rand_pool, BoidsData::YMIN, BoidsData::YMAX);
  Kokkos::fill_random(boidsData.dx, rand_pool, -1., 1.);
  Kokkos::fill_random(boidsData.dy, rand_pool, -1., 1.);

} // BoidsData::initPositions

// ===================================================
// ===================================================
/* std::pair<float,float> updateAverageVelocity(BoidsData& boidsData) */
/* { */

/*   // this is just a reduction */
/*   // we could also use a custom reducer, and place output directly in */
/*   // device memory */
/*   // see https://github.com/kokkos/kokkos/wiki/Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types */

/*   float vx, vy; */

/*   Kokkos::parallel_reduce("updateAverageVelocity x", boidsData.nBoids, */
/*      KOKKOS_LAMBDA(const int index, float& value) */
/*      { */
/*        value += boidsData.dx(index); */
/*      }, vx); */

/*   Kokkos::parallel_reduce("updateAverageVelocity y", boidsData.nBoids, */
/*      KOKKOS_LAMBDA(const int index, float& value) */
/*      { */
/*        value += boidsData.dy(index); */
/*      }, vy); */

/*   vx /= boidsData.nBoids; */
/*   vy /= boidsData.nBoids; */

/*   return std::make_pair(vx,vy); */

/* } // updateAverageVelocity */

// ===================================================
// ===================================================
void shuffleEnnemies(BoidsData& boidsData, MyRandomPool::RGPool_t& rand_pool, float rate)
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
      boidsData.ennemies(index) = Kokkos::rand<rnd_t,int>::draw(rand_gen, boidsData.nBoids);
    }

    // free random gen state, so that it can used by other threads later.
    rand_pool.free_state(rand_gen);

  });

} // BoidsData::shuffleEnnemies

// ===================================================
// ===================================================
void computeBoxData(BoidsData& boidsData)
{

  boidsData.resetBoxData();

  // compute the number of boids per box
  // and compute boids color
  // TODO:
  // - alternative : use Kokkos::ScatterView instead of Kokkos memory traits atomic

  using VecIntAtomic = BoidsData::VecIntAtomic;
  VecIntAtomic boxCount = boidsData.boxCount;

  using VecFloatAtomic = BoidsData::VecFloatAtomic;
  VecFloatAtomic box_x = boidsData.box_x;
  VecFloatAtomic box_y = boidsData.box_y;

  Kokkos::parallel_for("computeBoxCount",
                       boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    auto x = boidsData.x(index);
    auto y = boidsData.y(index);

    int iBox = pos2box(x,y);

    //printf("index=%d %f %f | iBox=%d | %d %d\n",index,x,y,iBox,pos2box<0>(x),pos2box<1>(y));

    boxCount(iBox) += 1;
    box_x(iBox) += x;
    box_y(iBox) += y;

    // update current boid color
    boidsData.color(index) = iBox;

  });

  Kokkos::parallel_for("compute box average velocity",
                       BoidsData::NBOX_X*BoidsData::NBOX_Y, KOKKOS_LAMBDA(const int& iBox)
  {
    auto n = boidsData.boxCount(iBox);
    if (n > 0)
    {
      boidsData.box_x(iBox) /= n;
      boidsData.box_y(iBox) /= n;
    }
    else
    {
      boidsData.box_x(iBox) = 0;
      boidsData.box_y(iBox) = 0;
    }
  });

  // sort boids per color
  auto permutation = kboids::sort(boidsData.color);

  // apply permutation to boids coordinates and displacements
  kboids::apply_permutation(boidsData.x,  boidsData.tmp, permutation);
  kboids::apply_permutation(boidsData.y,  boidsData.tmp, permutation);
  kboids::apply_permutation(boidsData.dx, boidsData.tmp, permutation);
  kboids::apply_permutation(boidsData.dy, boidsData.tmp, permutation);

   // for (int i = 0; i<100; ++i)
   //   printf("%d %d | perm=%d |%f %f %d %d\n",i,boidsData.color(i),permutation(i),boidsData.x(i),boidsData.y(i),
   //          pos2box<0>(boidsData.x(i)),
   //          pos2box<1>(boidsData.y(i)));


  // compute index to first boids of each color
  // using exclusive scan pattern
  Kokkos::parallel_scan("Compute BoxIndex", BoidsData::NBOX_X*BoidsData::NBOX_Y,
     KOKKOS_LAMBDA(const int iBox,
                   int& update, const bool final)
     {
       const int iTmp = boidsData.boxCount(iBox);

       if (final)
         boidsData.boxIndex(iBox) = update;

       update += iTmp;
     });

  //for (int i = 0; i<BoidsData::NBOX_X*BoidsData::NBOX_Y; ++i)
  //  printf("%d %d %d | %f %f\n",i,boidsData.boxCount(i),boidsData.boxIndex(i),boidsData.box_x(i),boidsData.box_y(i));

} // computeBoxData

// ===================================================
// ===================================================
void updatePositions(BoidsData& boidsData)
{

  // prepare data used in rule #2
  // i.e. adjust velocity to close neighbors
  computeBoxData(boidsData);

  const float centeringFactor = 0.0005;
  const float matchingFactor = 0.005;
  const float minDistance = 20;
  const float avoidFactor = 0.05;

  Kokkos::parallel_for(boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    //
    // rule #1 : flight towards center
    //
    const auto x = boidsData.x(index);
    const auto y = boidsData.y(index);

    float dx = boidsData.dx(index);
    float dy = boidsData.dy(index);

    float xc = (BoidsData::XMIN+BoidsData::XMAX)/2;
    float yc = (BoidsData::YMIN+BoidsData::YMAX)/2;
    dx += (xc-x) * centeringFactor;
    dy += (yc-y) * centeringFactor;

    //
    // rule #2, adjust velocity to move toward the gravity center of boids of same color
    //
    const auto color = boidsData.color(index);
    //const int nbColors = BoidsData.NBOX_X * BoidsData::NBOX_Y;

    //const auto box_dx = boidsData.box_dx(color);
    //const auto box_dy = boidsData.box_dy(color);
    const auto xg = boidsData.box_x(color);
    const auto yg = boidsData.box_y(color);
    dx += (xg-x) * matchingFactor;
    dy += (yg-y) * matchingFactor;

    //dx += matchingFactor * (box_dx - boidsData.dx(index));
    //dy += matchingFactor * (box_dy - boidsData.dy(index));

    /* float dx_n = 0; */
    /* float dy_n = 0; */
    /* { */
    /*   auto beg = boidsData.boxIndex(color); */
    /*   auto end = (color == nbColors-1) ? boidsData.nBoids : boidsData.boxIndex(color+1); */
    /*   for (int i_n=beg; i_n<end; ++i_n) */
    /*   { */
    /*   } */
    /* } */

    //
    // rule #3: avoid ennemy
    //
    auto index_ennemy = boidsData.ennemies(index);

    float dir_x, dir_y;

    compute_direction(x,y,
                      boidsData.x(index_ennemy),
                      boidsData.y(index_ennemy),
                      dir_x, dir_y);

    dx -= dir_x * avoidFactor;
    dy -= dir_y * avoidFactor;

    // speed limit
    speedLimit(dx,dy);

    keepInTheBox(x,y,dx,dy);

    // write final results
    boidsData.dx(index) = dx;
    boidsData.dy(index) = dy;

    // final update
    boidsData.x(index) += boidsData.dx(index);
    boidsData.y(index) += boidsData.dy(index);

  });

} // updatePositions

// ===================================================
// ===================================================
void copyPositionsForRendering(BoidsData& boidsData)
{

#ifdef FORGE_ENABLED

  Kokkos::parallel_for(boidsData.nBoids, KOKKOS_LAMBDA(const int& index)
  {
    auto x = boidsData.x(index);
    auto y = boidsData.y(index);

    boidsData.xy(2*index)   = (float)x;
    boidsData.xy(2*index+1) = (float)y;
  });

#endif

} // copyPositionsForRendering
