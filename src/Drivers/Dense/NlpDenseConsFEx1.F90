! =============================================================================
!
!     This example is modified from NlpSparseEx1
!
!
!     min   sum 1/4* { (x_{i}-1)^4 : i=1,...,n}
!     s.t.     4*x_1 + 2*x_2                   == 10
!          5<= 2*x_1         + x_3
!          1<= 2*x_1               + 0.5*x_i   <= 2*n, for i=4,...,n
!          x_1 free
!          0.0 <= x_2
!          1.5 <= x_3 <= 10
!          x_i >=0.5, i=4,...,n
! =============================================================================

program example1
#ifdef HIOP_USE_MPI
      USE MPI
#endif

! =============================================================================
!                main function
! =============================================================================
      implicit none

      integer :: ierr, num_procs, rank
      integer     N,     M
      integer     retv
      parameter  (N = 50, M = 49)

      double precision LAM(M)
      double precision G(M)
      double precision X(N)
      double precision X_L(N), X_U(N), Z_L(N), Z_U(N)
      double precision G_L(M), G_U(M)
      double precision OOBJ
      double precision OBJ_SAVED
      parameter (OBJ_SAVED = 1.10351564683176d-1)

      integer*8 HIOPPROBLEM
      integer*8 hiopdenseprob

      double precision F
      integer I

      ! Initialize MPI
#ifdef HIOP_USE_MPI
      write(*,*) "MPI is available"
      call MPI_Init(ierr)
      call MPI_Comm_size(MPI_COMM_WORLD, num_procs, ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
#else
      write(*,*) "MPI is not available"
      rank = 0
      num_procs = 1
#endif

!     Set initial point and bounds:
      do I = 1, N
        X(I) = 0d0
      enddo

      X_L(1) = -1d20
      X_U(1) =  1d20
      X_L(2) =  0d0
      X_U(2) =  1d20
      X_L(3) =  1.5d0
      X_U(3) =  10d0

      do I = 4, N
        X_L(I) = 0.5d0
        X_U(I) = 1d20
      enddo

!     Set bounds for the constraints
      G_L(1) =  1d1
      G_U(1) =  1d1
      G_L(2) =  5d0
      G_U(2) =  1d20

      do I = 3, M
        G_L(I) = 1d0
        G_U(I) = 2d0*N
      enddo

!     create hiop sparse problem
      HIOPPROBLEM = hiopdenseprob(N, M,X_L, X_U, G_L, G_U, X, &
                                  EVAL_F, EVAL_CON, EVAL_GRAD, EVAL_JAC)
      if (HIOPPROBLEM.eq.0) then
         write(*,*) 'Error creating an HIOP Problem handle.'
         stop
      endif

!     hiop solve 
      call hiopdensesolve(HIOPPROBLEM,OOBJ,X)

      write(*,*)
      write(*,*) 'The optimal solution is:'
      write(*,*)
      write(*,*) 'Optimal Objective = ',OOBJ
      write(*,*)

      if (ABS(OOBJ-OBJ_SAVED) > 1e-6) then
         write(*,*) 'Obj mismatches SparseEx1 with 50 variables.'
         write(*,*) 'Saved Obj = ', OBJ_SAVED 
         stop -1
      endif

#ifdef HIOP_USE_MPI
      ! Clean up MPI
      call MPI_Finalize(ierr)
#endif

!     Clean up
      call deletehiopdenseprob(HIOPPROBLEM)
      stop



contains
! =============================================================================
!                    Computation of objective function
! =============================================================================
      subroutine EVAL_F(N, X, NEW_X, OBJ)
      implicit none
      integer N, NEW_X, I
      double precision OBJ, X(N)
      OBJ = 0
      do I = 1, N
         OBJ = OBJ + 0.25d0*(X(I)-1d0)**4
      enddo
      return
      end

! =============================================================================
!                Computation of gradient of objective function
! =============================================================================
      subroutine EVAL_GRAD(N, X, NEW_X, GRAD)
      implicit none
      integer N, NEW_X, I
      double precision GRAD(N), X(N)
      do I = 1, N
         GRAD(I) = (X(I)-1d0)**3
      enddo
      return
      end

! =============================================================================
!                     Computation of equality constraints
! =============================================================================
      subroutine EVAL_CON(N, M, X, NEW_X, C)
      implicit none
      integer N, NEW_X, M, I
      double precision C(M), X(N)
      C(1) = 4d0*X(1) + 2d0*X(2)
      C(2) = 2d0*X(1) + 1d0*X(3)
      do I = 3, M
         C(I) = 2d0*X(1) + 0.5d0*X(I+1)
      enddo
      return
      end

! =============================================================================
!                Computation of Jacobian of equality constraints
! =============================================================================

      subroutine EVAL_JAC(N, M, X, NEW_X, A)
      integer N, NEW_X, M, NZ
      double precision X(N), A(N*M)
      integer I, CONIDX

      NZ = N*M
      do I = 1, NZ
        A(I) = 0d0
      enddo

!     // constraint 1 body ---> 4*x_1 + 2*x_2 == 10
!     // constraint 2 body ---> 2*x_1 + x_3
!     // constraint 3 body ---> 2*x_1 + 0.5*x_i, for i>=4
      A(1) = 4d0
      A(2) = 2d0
      A(N+1) = 2d0
      A(N+3) = 1d0

      do I = 3, M
        A((I-1)*N+1) = 2d0
        A((I-1)*N+I+1) = 0.5d0
      enddo

      return
      end


end program example1