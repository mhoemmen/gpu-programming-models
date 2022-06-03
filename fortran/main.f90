! Based on https://github.com/kokkos/kokkos-fortran-interop/tree/develop/examples/03-axpy-view

program summation
    use :: flcl_mod
    use :: flcl_util_kokkos_mod
    use :: axpy_f_mod

    implicit none
    
    integer, parameter :: n = 100000000
    integer :: i
    real(REAL64), pointer, dimension(:)  :: x, y, z
    type(view_r64_1d_t) :: v_x, v_y, v_z

    ! Initialize kokkos
    call kokkos_initialize()

    ! Allocate views
    call kokkos_allocate_view( x, v_x, 'x', int(n, c_size_t) )
    call kokkos_allocate_view( y, v_y, 'y', int(n, c_size_t) )
    call kokkos_allocate_view( z, v_z, 'z', int(n, c_size_t) )

    ! Initialize x and y
    do i = 1, n
        x(i) = i;
        y(i) = n - i;
    end do

    ! Compute sum of x and y
    call axpy_view(v_x, v_y, v_z)

    ! Assert that the sum is correct
    do i = 1, n
        if (z(i) .ne. n) then
            print *, "incorrect sum", i, x(i), y(i), z(i), n
            call exit(1)
        end if
    end do
    
    ! Deallocate views
    call kokkos_deallocate_view( x, v_x )
    call kokkos_deallocate_view( y, v_y )
    call kokkos_deallocate_view( z, v_z )

    ! Finalize kokkos
    call kokkos_finalize()
end
