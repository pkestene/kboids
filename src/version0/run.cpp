#include "Boids.h"

#include <iostream>
#include <cstdint>

#include "utils/likwid-utils.h"
#include "time/Timer.h"

// =====================================================================================
// =====================================================================================
void run_boids_flight(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data)
{

  // create a BoidsData object
  BoidsData boidsData(nBoids);

  // init friends and ennemies
  MyRandomPool myRandPool(seed);

  initPositions(boidsData, myRandPool.pool);
  shuffleFriendsAndEnnemies(boidsData, myRandPool.pool);

  // 2d array for display
  PngData data("render_image", 768, 768, 4);

  Timer timer;

  for(int iTime=0; iTime<nIter; ++iTime)
  {

    timer.start();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      LIKWID_MARKER_START("updatePositions");
    }

    updatePositions(boidsData);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      LIKWID_MARKER_STOP("updatePositions");
    }

    if (iTime % 20 == 0)
      shuffleFriendsAndEnnemies(boidsData, myRandPool.pool);
    timer.stop();

    // should we dump data to file ?
    if (dump_data and (iTime % 1 == 0))
    {
      std::cout << "Save data at time step : " << iTime << "\n";
      renderPositions(data, boidsData);
      savePositions("boids",iTime,data);
    }

  } // end for iTimer

  // report time spent in computations
  auto time_seconds = timer.elapsed();
  std::cout << "Total time : " << time_seconds << " seconds\n";
  std::cout << "Throughput : " << (nBoids*nIter)/time_seconds/1e6 << " MBoids-updates/s \n";

} // run_boids_flight
