#include "Boids.h"

#include <iostream>
#include <cstdint>

#ifdef KOKKOS_ENABLE_CUDA
#include "time/CudaTimer.h"
using Timer = CudaTimer;
#elif defined(KOKKOS_ENABLE_OPENMP)
#include "time/OpenMPTimer.h"
using Timer = OpenMPTimer;
#else
#include "time/SimpleTimer.h"
using Timer = SimpleTimer;
#endif


void run_boids_dance(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data)
{

  // create a Dance object
  Dance dance(nBoids);

  // init friends and ennemies
  MyRandomPool myRandPool(seed);

  initPositions(dance, myRandPool.pool);
  shuffleFriendsAndEnnemies(dance, myRandPool.pool);

  // 2d array for display
  PngData data("render_image", 768, 768, 4);

  Timer timer;

  for(int iTime=0; iTime<nIter; ++iTime)
  {

    timer.start();
    updatePositions(dance);

    if (iTime % 20 == 0)
      shuffleFriendsAndEnnemies(dance, myRandPool.pool);
    timer.stop();

    // should we dump data to file ?
    if (dump_data and (iTime % 1 == 0))
    {
      std::cout << "Save data at time step : " << iTime << "\n";
      renderPositions(data, dance);
      savePositions("boids",iTime,data);
    }

  } // end for iTimer

  // report time spent in computations
  auto time_seconds = timer.elapsed();
  std::cout << "Total time : " << time_seconds << " seconds\n";
  std::cout << "Throughput : " << (nBoids*nIter)/time_seconds << " Boids-updates/s \n";

} // run_boids_dance
