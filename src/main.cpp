#include "Boids.h"
#include "run.h"

#include <iostream>
#include <argh.h>


// Include Kokkos Headers
#include<Kokkos_Core.hpp>

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
      "  -g, --gui               add a simple visualization gui (require FORGE library)\n"
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
int main(int argc, char* argv[])
{

  argh::parser cmdl({
      "-n", "--nboids",
      "-i", "--iter",
      "-s", "--seed",
      "-g", "--gui"});
    cmdl.parse(argc, argv);


  int nBoids;
  cmdl({"n", "nboids"}, 1000) >> nBoids;

  uint32_t nIter;
  cmdl({"i", "iter"}, 100) >> nIter;

  // initialize the random generator pool
  //uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  uint64_t seed;
  cmdl({"s", "seed"}, 42) >> seed;

  bool guiEnabled = cmdl[{"g","--gui"}];
  if (guiEnabled)
  {
    std::cout << "GUI enabled, we limit the number of boids to 1000\n";
    nBoids = (nBoids > 1000) ? 1000 : nBoids;
  }

  if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
  }

  // =================================================

  //Initialize Kokkos
  Kokkos::initialize(argc,argv);

  {
    print_kokkos_config();

    run_boids_dance(nBoids, nIter, seed);

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;

} // main
