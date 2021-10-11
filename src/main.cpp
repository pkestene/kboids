#include <iostream>

#include "Boids.h"
#include "run.h"

// Include Kokkos Headers
#include<Kokkos_Core.hpp>

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

  int nBoids     = (argc>1) ? std::atoi(argv[1]) : 1000;
  uint32_t nIter = (argc>2) ? std::atoi(argv[2]) : 1000;
  uint64_t seed  = 42;

  if (argc>3)
  {
    seed = std::atoi(argv[3]);
  }
  else
  {
    // initialize the random generator pool
    uint64_t ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    std::cout << "[" << __FUNCTION__ << "]:"
              << " Using random seed: " << ticks << std::endl;
  }


  //Initialize Kokkos
  Kokkos::initialize(argc,argv);

  {
    print_kokkos_config();

    run_boids_dance(nBoids, nIter, seed);

  }

  Kokkos::finalize();

} // main
