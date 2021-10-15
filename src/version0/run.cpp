#include "Boids.h"

#include <iostream>
#include <cstdint>

#include "utils/likwid-utils.h"
#include "time/Timer.h"

#ifdef FORGE_ENABLED
#include <forge.h>
#ifdef KOKKOS_ENABLE_CUDA
#define USE_FORGE_CUDA_COPY_HELPERS
#else
#define USE_FORGE_CPU_COPY_HELPERS
#endif
#include <fg/compute_copy.h>
#endif


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

#ifdef FORGE_ENABLED
// =====================================================================================
// =====================================================================================
void run_boids_flight_gui(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data)
{

  // create a BoidsData object
  BoidsData boidsData(nBoids);

  // init friends and ennemies
  MyRandomPool myRandPool(seed);

  initPositions(boidsData, myRandPool.pool);
  shuffleFriendsAndEnnemies(boidsData, myRandPool.pool);

  // Forge init
  const int DIMX=800;
  const int DIMY=800;
  forge::Window wnd(DIMX, DIMY, "Boids flight version 0");
  wnd.makeCurrent();

  forge::Chart chart(FG_CHART_2D);
  chart.setAxesLimits(-1.05f, 1.05f, -1.05f, 1.05f);

  forge::Plot boidsXY =
    chart.plot(nBoids, forge::f32, FG_PLOT_SCATTER, FG_MARKER_CIRCLE);
  boidsXY.setColor(0.1f, 0.1f, 0.2f, 1.f);
  boidsXY.setMarkerSize(6);

  GfxHandle *handles;

  createGLBuffer(&handles, boidsXY.vertices(), FORGE_VERTEX_BUFFER);

  int iTime = 0;
  do {

    updatePositions(boidsData);
    copyPositionsForRendering(boidsData);

    if (iTime % 20 == 0)
      shuffleFriendsAndEnnemies(boidsData, myRandPool.pool);

    copyToGLBuffer(handles, (ComputeResourceHandle)boidsData.xy.data(),
                   boidsXY.verticesSize());
    wnd.draw(chart);

    iTime++;

  } while (!wnd.close());

  // destroy GL-cpu interop buffers
  releaseGLBuffer(handles);

} // run_boids_flight_gui
#endif
