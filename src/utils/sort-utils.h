#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

/*
 * The following is borrowed from ArborX :
 * https://github.com/arborx/ArborX
 */

/** \brief Fills the view with a sequence of numbers
 *
 *  \param[in] space Execution space
 *  \param[out] v Output view
 *  \param[in] value (optional) Initial value
 *
 *  \note Similar to \c std::iota() but differs in that it directly assigns
 *  <code>v(i) = value + i</code> instead of repetitively evaluating
 *  <code>++value</code> which would be difficult to achieve in a performant
 *  manner while still guaranteeing the order of execution.
 */

namespace kboids {

//===============================================================================
//===============================================================================
template <typename ExecutionSpace, typename T, typename... P>
void iota(ExecutionSpace &&space, Kokkos::View<T, P...> const &v,
          typename Kokkos::ViewTraits<T, P...>::value_type value = 0)
{
  using ValueType = typename Kokkos::ViewTraits<T, P...>::value_type;
  static_assert((unsigned(Kokkos::ViewTraits<T, P...>::rank) == unsigned(1)),
                "iota requires a View of rank 1");
  static_assert(std::is_arithmetic<ValueType>::value,
                "iota requires a View with an arithmetic value type");
  static_assert(
      std::is_same<ValueType, typename Kokkos::ViewTraits<
                                  T, P...>::non_const_value_type>::value,
      "iota requires a View with non-const value type");
  auto const n = v.extent(0);
  Kokkos::RangePolicy<std::decay_t<ExecutionSpace>> policy(
      std::forward<ExecutionSpace>(space), 0, n);
  Kokkos::parallel_for("ArborX::Algorithms::iota", policy,
                       KOKKOS_LAMBDA(int i) { v(i) = value + (ValueType)i; });
}

//===============================================================================
//===============================================================================
template <class ViewType,
          class SizeType = unsigned int>
Kokkos::View<SizeType *, typename ViewType::device_type>
sort(ViewType view)
{
  static_assert(ViewType::rank == 1, "Only sorting a View of rank 1");

  int const n = view.extent(0);

  using range_policy = Kokkos::RangePolicy<typename ViewType::execution_space>;
  using ValueType    = typename ViewType::value_type;
  using CompType     = Kokkos::BinOp1D<ViewType>;

  Kokkos::MinMaxScalar<ValueType> result;
  Kokkos::MinMax<ValueType> reducer(result);

  parallel_reduce("Kokkos sort find min/max of view",
                  range_policy(0, n),
                  Kokkos::Impl::min_max_functor<ViewType>(view),
                  reducer);

  // data are already sorted, returning identity
  if (result.min_val == result.max_val)
  {
    // allocating memory for permutation view
    Kokkos::View<SizeType *, typename ViewType::device_type> permute(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "permute"), n);
    iota(Kokkos::DefaultExecutionSpace{}, permute);
    return permute;
  }

  Kokkos::BinSort<ViewType, CompType, typename ViewType::device_type, SizeType> bin_sort(
    view,
    CompType(n / 2, result.min_val, result.max_val), true);

  bin_sort.create_permute_vector();
  bin_sort.sort(view);

  return bin_sort.get_permute_vector();
}

//===============================================================================
//===============================================================================
template <class ViewType,
          class SizeType = unsigned int>
void apply_permutation(ViewType view,
                       ViewType view_tmp,
                       Kokkos::View<SizeType *, typename ViewType::device_type> permutation)
{
  int const n = view.extent(0);

  Kokkos::parallel_for("Apply permutation", n,
    KOKKOS_LAMBDA(const int index)
    {
      view_tmp(index) = view(permutation(index));
    });

  std::swap(view, view_tmp);

}



} // namespace kboids
