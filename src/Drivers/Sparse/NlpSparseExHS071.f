C =============================================================================
C
C     This example is modified from IPOPT hs071_f.f.
C
C     This is an example for the usage of HIOP in double precision.
C     It implements problem 71 from the Hock-Schittkowski test suite:
C
C     min   x1*x4*(x1 + x2 + x3)  +  x3
C     s.t.  x1*x2*x3*x4                   >=  25
C           x1**2 + x2**2 + x3**2 + x4**2  =  40
C           1 <=  x1,x2,x3,x4  <= 5
C
C     Starting point:
C        x = (1, 5, 5, 1)
C
C     Optimal solution:
C        x = (1.00000000, 4.74299963, 3.82114998, 1.37940829)
C
C =============================================================================

      program example
      implicit none

      integer     N,     M,     NELE_JACEQ,     NELE_JACIN
      integer     NELE_HESS, retv
      parameter  (N = 4, M = 2, NELE_JACEQ = 4, NELE_JACIN = 4)
      parameter  (NELE_HESS = 10 )

      double precision LAM(M)
      double precision G(M)
      double precision X(N)
      double precision X_L(N), X_U(N), Z_L(N), Z_U(N)
      double precision G_L(M), G_U(M)
      double precision OOBJ

      integer*8 HIOPPROBLEM
      integer*8 hiopsparseprob

      integer IPSOLVE

      double precision F
      integer i

      external EVAL_F, EVAL_CON, EVAL_GRAD, EVAL_JAC, EVAL_HESS

C     Set initial point and bounds:
      data X   / 1d0, 5d0, 5d0, 1d0/
      data X_L / 1d0, 1d0, 1d0, 1d0 /
      data X_U / 5d0, 5d0, 5d0, 5d0 /

C     Set bounds for the constraints
      data G_L / 25d0, 40d0 /
      data G_U / 1d40, 40d0 /

C     create hiop sparse problem
      HIOPPROBLEM = hiopsparseprob(N, M, NELE_JACEQ,
     1     NELE_JACIN, NELE_HESS,X_L, X_U, G_L, G_U, X,
     1     EVAL_F, EVAL_CON, EVAL_GRAD, EVAL_JAC, EVAL_HESS)
      if (HIOPPROBLEM.eq.0) then
         write(*,*) 'Error creating an HIOP Problem handle.'
         stop
      endif

C     hiop solve 
      call hiopsparsesolve(HIOPPROBLEM,OOBJ,X)

      write(*,*)
      write(*,*) 'The optimal solution is:'
      write(*,*)
      write(*,*) 'Optimal Objective = ',OOBJ
      write(*,*)
      write(*,*) 'The optimal values of X are:'
      write(*,*)
      do i = 1, N
         write(*,*) 'X  (',i,') = ',X(i)
      enddo
      write(*,*)

C     Clean up
      call deletehiopsparseprob(HIOPPROBLEM)
      stop
      end

C =============================================================================
C                    Computation of objective function
C =============================================================================
      subroutine EVAL_F(N, X, NEW_X, OBJ)
      implicit none
      integer N, NEW_X
      double precision OBJ, X(N)
      OBJ = X(1)*X(4)*(X(1)+X(2)+X(3)) + X(3)
      return
      end

C =============================================================================
C                Computation of gradient of objective function
C =============================================================================
      subroutine EVAL_GRAD(N, X, NEW_X, GRAD)
      implicit none
      integer N, NEW_X
      double precision GRAD(N), X(N)
      GRAD(1) = X(4)*(2d0*X(1)+X(2)+X(3))
      GRAD(2) = X(1)*X(4)
      GRAD(3) = X(1)*X(4) + 1d0
      GRAD(4) = X(1)*(X(1)+X(2)+X(3))
      return
      end

C =============================================================================
C                     Computation of equality constraints
C =============================================================================
      subroutine EVAL_CON(N, M, X, NEW_X, C)
      implicit none
      integer N, NEW_X, M
      double precision C(M), X(N)
      C(1) = X(1)*X(2)*X(3)*X(4) 
      C(2) = X(1)**2 + X(2)**2 + X(3)**2 + X(4)**2
      return
      end

C =============================================================================
C                Computation of Jacobian of equality constraints
C =============================================================================

      subroutine EVAL_JAC(TASK, N, M, X, NEW_X, NZ, IROW, JCOL, A)
      integer TASK, N, NEW_X, M, NZ
      double precision X(N), A(NZ)
      integer IROW(NZ), JCOL(NZ), I

C     structure of Jacobian:
      integer AVAR1(8), ACON1(8)
      data  AVAR1 /1, 2, 3, 4, 1, 2, 3, 4/
      data  ACON1 /1, 1, 1, 1, 2, 2, 2, 2/
      save  AVAR1, ACON1

      if( TASK.eq.0 ) then
        do I = 1, 8
          JCOL(I) = AVAR1(I)
          IROW(I) = ACON1(I)
        enddo
      else
        A(1) = X(2)*X(3)*X(4)
        A(2) = X(1)*X(3)*X(4)
        A(3) = X(1)*X(2)*X(4)
        A(4) = X(1)*X(2)*X(3)
        A(5) = 2d0*X(1)
        A(6) = 2d0*X(2)
        A(7) = 2d0*X(3)
        A(8) = 2d0*X(4)
      endif
      return
      end

C =============================================================================
C                Computation of Hessian of Lagrangian
C =============================================================================
      subroutine EVAL_HESS(TASK, N, M, OBJFACT, X, NEW_X, LAM, NEW_LAM,
     1     NNZH, IROW, JCOL, HESS)
      implicit none
      integer TASK, N, NEW_X, M, NEW_LAM, NNZH, i
      double precision X(N), OBJFACT, LAM(M), HESS(NNZH)
      integer IROW(NNZH), JCOL(NNZH)

C     structure of Hessian:
      integer IRNH1(10), ICNH1(10)
      data  IRNH1 /1, 2, 2, 3, 3, 3, 4, 4, 4, 4/
      data  ICNH1 /1, 1, 2, 1, 2, 3, 1, 2, 3, 4/
      save  IRNH1, ICNH1

      if( TASK.eq.0 ) then
         do i = 1, 10
            IROW(i) = IRNH1(i)
            JCOL(i) = ICNH1(i)
         enddo
      else
         do i = 1, 10
            HESS(i) = 0d0
         enddo

C     objective function
         HESS(1) = OBJFACT * 2d0*X(4)
         HESS(2) = OBJFACT * X(4)
         HESS(4) = OBJFACT * X(4)
         HESS(7) = OBJFACT * (2d0*X(1) + X(2) + X(3))
         HESS(8) = OBJFACT * X(1)
         HESS(9) = OBJFACT * X(1)

C     first constraint
         HESS(2) = HESS(2) + LAM(1) * X(3)*X(4)
         HESS(4) = HESS(4) + LAM(1) * X(2)*X(4)
         HESS(5) = HESS(5) + LAM(1) * X(1)*X(4)
         HESS(7) = HESS(7) + LAM(1) * X(2)*X(3)
         HESS(8) = HESS(8) + LAM(1) * X(1)*X(3)
         HESS(9) = HESS(9) + LAM(1) * X(1)*X(2)

C     second constraint
         HESS(1) = HESS(1) + LAM(2) * 2d0
         HESS(3) = HESS(3) + LAM(2) * 2d0
         HESS(6) = HESS(6) + LAM(2) * 2d0
         HESS(10)= HESS(10)+ LAM(2) * 2d0
      endif
      return
      end

