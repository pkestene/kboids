#pragma once

void run_boids_flight(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data);

#ifdef FORGE_ENABLED
void run_boids_flight_gui(uint32_t nBoids, uint32_t nIter, uint64_t seed, bool dump_data);
#endif
