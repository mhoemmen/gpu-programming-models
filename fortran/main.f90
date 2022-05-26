program summation
    implicit none
    integer, parameter :: n = 1e8
    integer :: i
    real, dimension(:), allocatable :: x, y, z

    allocate(x(n), y(n), z(n))

    ! Initialize x and y
    do i = 1, n
        x(i) = i;
        y(i) = n - i;
    end do

    ! Compute sum of x and y
    do i = 1, n
        z(i) = x(i) + y(i)
    end do

    ! Assert that the sum is correct
    do i = 1, n
        if (z(i) .ne. n) then
            print *, "incorrect sum"
            call exit(1)
        end if
    end do
    
    deallocate(x, y, z)
end
