#include "Boids.h"
#include "run.h"
#include "run_gui.h"

#include <iostream>
#include <argh.h>

// Include Kokkos Headers
#include<Kokkos_Core.hpp>

#include "utils/likwid-utils.h"

// ===================================================
// ===================================================
void usage(const std::string &app_name)
{
    std::string msg =
      app_name +
      " miniapp\n"
      "\n"
      "Usage:\n"
      "  " +
      app_name +
      " [OPTION...]\n"
      "  -n, --nboids arg        Number of boids (default: 1000)\n"
      "  -i, --iter arg          Number of time steps (default: 100)\n"
      "  -s, --seed arg          Random seed (default: 42)\n"
      "  -d, --dump              Dump data to PNG files\n"
      "  -g, --gui               Add a simple visualization gui (require FORGE library)\n"
      "  -h, --help              Show this help";

      std::cout << msg << std::endl;
}

// ===================================================
// ===================================================
void print_kokkos_config()
{

  std::cout << "##########################\n";
  std::cout << "KOKKOS CONFIG             \n";
  std::cout << "##########################\n";

  std::ostringstream msg;
  std::cout << "Kokkos configuration" << std::endl;
  if ( Kokkos::hwloc::available() ) {
    msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
    << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
    << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
    << "] )"
    << std::endl ;
  }
  Kokkos::print_configuration( msg );
  std::cout << msg.str();
  std::cout << "##########################\n";

} // print_kokkos_config

// ===================================================
// ===================================================
template<typename DeviceType>
void run_simu(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data, bool gui_enabled)
{
  if (gui_enabled)
    {
#ifdef FORGE_ENABLED
      run_boids_flight_gui<DeviceType>(nBoids, nIter, seed, dump_data);
#else
      std::cerr << "Rerun cmake and enable Forge library.\n";
#endif
    }
    else
    {

      LIKWID_MARKER_INIT;

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        LIKWID_MARKER_THREADINIT;
        LIKWID_MARKER_REGISTER("updatePositions");
      }

      // std::cout << "GUI enabled, we limit the number of boids to 1000\n";
      // nBoids = (nBoids > 1000) ? 1000 : nBoids;

      run_boids_flight<DeviceType>(nBoids, nIter, seed, dump_data);

      LIKWID_MARKER_CLOSE;

    }

}

// ===================================================
// ===================================================
int main(int argc, char* argv[])
{

  argh::parser cmdl({
      "-n", "--nboids",
      "-i", "--iter",
      "-s", "--seed",
      "-d", "--dump",
      "-g", "--gui",
      "-c", "--cuda"});
    cmdl.parse(argc, argv);


  uint32_t nBoids;
  cmdl({"n", "nboids"}, 1000) >> nBoids;

  uint32_t nIter;
  cmdl({"i", "iter"}, 100) >> nIter;

  // initialize the random generator pool
  //uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  uint64_t seed;
  cmdl({"s", "seed"}, 42) >> seed;

  bool dump_data = cmdl[{"d","dump"}];

  bool gui_enabled = cmdl[{"g","gui"}];

  bool use_cuda_version = cmdl[{"c", "cuda"}];

  if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
  }

  // =================================================

  //Initialize Kokkos
  Kokkos::initialize(argc,argv);

  {
    print_kokkos_config();

#ifdef KOKKOS_ENABLE_OPENMP
    std::cout << "=================================\n";
    std::cout << "Running on Kokkos::OpenMP device \n";
    std::cout << "=================================\n";
    run_simu<Kokkos::OpenMP>(nBoids, nIter, seed, dump_data, false);
#endif

#ifdef KOKKOS_ENABLE_CUDA
    std::cout << "=================================\n";
    std::cout << "Running on Kokkos::Cuda device   \n";
    std::cout << "=================================\n";
    run_simu<Kokkos::Cuda>(nBoids, nIter, seed, dump_data, false);
#endif
  }

  Kokkos::finalize();

  return EXIT_SUCCESS;

} // main
