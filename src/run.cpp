#include "Boids.h"

#include <iostream>
#include <cstdint>

void run_boids_dance(uint32_t nBoids, uint32_t nIter, uint64_t seed)
{

  // create a Dance object
  Dance dance(nBoids);

  // init friends and ennemies
  MyRandomPool myRandPool(seed);

  initPositions(dance, myRandPool.pool);
  shuffleFriendsAndEnnemies(dance, myRandPool.pool);

  // for (int i=0; i<dance.nBoids; ++i)
  // {
  //   std::cout << " friends and ennemies : (" << i << ") " 
  //             << dance.friends(i) << " "
  //             << dance.ennemies(i) << "\n";
  // }

  // 2d array for display
  PngData data("render_image", 768, 768, 4);

  for(int iTime=0; iTime<nIter; ++iTime)
  {

    updatePositions(dance);

    if (iTime % 20 == 0)
      shuffleFriendsAndEnnemies(dance, myRandPool.pool);

    if (iTime % 1 == 0)
    {
      std::cout << "Save data at time step : " << iTime << "\n";
      renderPositions(data, dance);
      savePositions("boids",iTime,data);
    }

  }


} // run_boids_dance
