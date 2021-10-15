#ifndef TIME_TIMER_H_
#define TIME_TIMER_H_

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

#endif /* TIME_TIMER_H */
