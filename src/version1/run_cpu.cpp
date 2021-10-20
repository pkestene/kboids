#include "run_gui.h"

#ifdef FORGE_ENABLED

#include <forge.h>
#define USE_FORGE_CPU_COPY_HELPERS
#include <fg/compute_copy.h>

// ===================================================================================
// ===================================================================================
template<>
void run_boids_flight_gui<Kokkos::OpenMP>(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data)
{

  using execution_space = Kokkos::OpenMP::execution_space;

  execution_space exec_space{};

  // create a BoidsData object
  BoidsData<Kokkos::OpenMP> boidsData(nBoids, seed);

  boidsData.initPositions();
  boidsData.shuffleFriendsAndEnnemies(exec_space, 1.0);

  // Forge init
  const int DIMX=800;
  const int DIMY=800;
  forge::Window wnd(DIMX, DIMY, "Boids flight version 0");
  wnd.makeCurrent();

  forge::Chart chart(FG_CHART_2D);
  chart.setAxesLimits(-2.05f, 2.05f, -2.05f, 2.05f);

  forge::Plot boidsXY =
    chart.plot(nBoids, forge::f32, FG_PLOT_SCATTER, FG_MARKER_CIRCLE);
  boidsXY.setColor(0.1f, 0.1f, 0.2f, 1.f);
  boidsXY.setMarkerSize(6);

  GfxHandle *handles;

  createGLBuffer(&handles, boidsXY.vertices(), FORGE_VERTEX_BUFFER);

  int iTime = 0;
  do {

    boidsData.updatePositions(exec_space);
    boidsData.copyPositionsForOpenGLRendering(exec_space);

    if (iTime % 20 == 0)
      boidsData.shuffleFriendsAndEnnemies(exec_space, 0.1);

    copyToGLBuffer(handles, (ComputeResourceHandle)boidsData.m_xy.data(),
                   boidsXY.verticesSize());
    wnd.draw(chart);

    iTime++;

  } while (!wnd.close());

  // destroy GL-cpu interop buffers
  releaseGLBuffer(handles);

} // run_boids_flight_gui

#endif // FORGE_ENABLED
