module axpy_f_mod
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env
  
    use :: flcl_mod
  
    implicit none
  
    public

      interface
        subroutine f_axpy_view( x, y, z ) &
          & bind(c, name='c_axpy_view')
          use, intrinsic :: iso_c_binding
          use :: flcl_mod
          type(c_ptr), intent(in) :: x, y, z
        end subroutine f_axpy_view
      end interface

      contains

        subroutine axpy_view( x, y, z )
          use, intrinsic :: iso_c_binding
          use :: flcl_mod
          implicit none
          type(view_r64_1d_t), intent(inout) :: z
          type(view_r64_1d_t), intent(in) :: x, y

          call f_axpy_view(x%ptr(), y%ptr(), z%ptr())

        end subroutine axpy_view
  
end module axpy_f_mod