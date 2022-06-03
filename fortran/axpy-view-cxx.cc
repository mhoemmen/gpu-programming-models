#include <Kokkos_Core.hpp>
#include "flcl-cxx.hpp"

using view_type = flcl::view_r64_1d_t;

extern "C" {

  void c_axpy_view( view_type **v_x, view_type **v_y, view_type **v_z ) {
    using flcl::view_from_ndarray;

    view_type x = **v_x;
    view_type y = **v_y;
    view_type z = **v_z;

    Kokkos::parallel_for( "axpy", z.extent(0), KOKKOS_LAMBDA( const size_t idx)
    {
      z(idx) = x(idx) + y(idx);
    });

    return;
  }

}