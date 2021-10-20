#include "io/lodepng.h"

#include <chrono>
#include <iostream>
#include <sstream>

// Include Kokkos Headers
#include<Kokkos_Core.hpp>

using PngData = Kokkos::View<unsigned char***, Kokkos::LayoutRight, Kokkos::Serial>;

// ===================================================
// ===================================================
void savePositions(std::string prefix, int time, PngData data)
{
  std::ostringstream outNum;
  outNum.width(7);
  outNum.fill('0');
  outNum << time;

  std::string filename = prefix + "_" + outNum.str() + ".png";

  lodepng_encode32_file(filename.c_str(), data.data(), data.extent(0), data.extent(1));

}
