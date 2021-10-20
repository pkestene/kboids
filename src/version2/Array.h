#pragma once

// borrowed from
// https://github.com/kokkos/kokkos/wiki/Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types

template<class Scalar_t, int N>
struct Array_t
{
  Scalar_t data[N];

  KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
  Array_t()
  {
    for (int i = 0; i < N; ++i )
    {
      data[i] = 0;
    }
  }

  KOKKOS_INLINE_FUNCTION   // Copy Constructor
  Array_t(const Array_t & rhs)
  {
    for (int i = 0; i < N; ++i )
    {
      data[i] = rhs.data[i];
    }
  }

  KOKKOS_INLINE_FUNCTION   // add operator
  Array_t& operator += (const Array_t& src)
  {
    for ( int i = 0; i < N; ++i )
    {
      data[i] += src.data[i];
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION   // volatile add operator
  void operator += (const volatile Array_t& src) volatile
  {
    for ( int i = 0; i < N; ++i )
    {
      data[i] += src.data[i];
    }
  }
};
