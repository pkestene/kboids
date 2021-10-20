#pragma once

#include "Boids.h"

#include <iostream>
#include <cstdint>

#include "utils/likwid-utils.h"
#include "time/Timer.h"

#ifdef FORGE_ENABLED

// ===================================================================================
// ===================================================================================
template<typename DeviceType>
void run_boids_flight_gui(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data)
{
}  // run_boids_flight_gui

template<>
void run_boids_flight_gui<Kokkos::OpenMP>(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data);

template<>
void run_boids_flight_gui<Kokkos::Cuda>(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data);

#endif // FORGE_ENABLED
