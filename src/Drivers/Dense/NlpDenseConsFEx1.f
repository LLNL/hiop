C =============================================================================
C
C     This example is modified from NlpSparseEx1
C
C
C     min   sum 1/4* { (x_{i}-1)^4 : i=1,...,n}
C     s.t.     4*x_1 + 2*x_2                   == 10
C          5<= 2*x_1         + x_3
C          1<= 2*x_1               + 0.5*x_i   <= 2*n, for i=4,...,n
C          x_1 free
C          0.0 <= x_2
C          1.5 <= x_3 <= 10
C          x_i >=0.5, i=4,...,n
C =============================================================================

      program example1
      implicit none

      integer     N,     M,
      integer     retv
      parameter  (N = 500, M = 499)

      double precision LAM(M)
      double precision G(M)
      double precision X(N)
      double precision X_L(N), X_U(N), Z_L(N), Z_U(N)
      double precision G_L(M), G_U(M)
      double precision OOBJ

      integer*8 HIOPPROBLEM
      integer*8 hiopdenseprob

      double precision F
      integer i

      external EVAL_F, EVAL_CON, EVAL_GRAD, EVAL_JAC

C     Set initial point and bounds:
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

C     Set bounds for the constraints
      G_L(1) =  1d1
      G_U(1) =  1d1
      G_L(2) =  5d0
      G_U(2) =  1d20

      do I = 3, M
        G_L(I) = 1d0
        G_U(I) = 2d0*N
      enddo

C     create hiop sparse problem
      HIOPPROBLEM = hiopdenseprob(N, M,X_L, X_U, G_L, G_U, X,
     1     EVAL_F, EVAL_CON, EVAL_GRAD, EVAL_JAC)
      if (HIOPPROBLEM.eq.0) then
         write(*,*) 'Error creating an HIOP Problem handle.'
         stop
      endif

C     hiop solve 
      call hiopdensesolve(HIOPPROBLEM,OOBJ,X)

      write(*,*)
      write(*,*) 'The optimal solution is:'
      write(*,*)
      write(*,*) 'Optimal Objective = ',OOBJ
      write(*,*)

C     Clean up
      call deletehiopdenseprob(HIOPPROBLEM)
      stop
      end

C =============================================================================
C                    Computation of objective function
C =============================================================================
      subroutine EVAL_F(N, X, NEW_X, OBJ)
      implicit none
      integer N, NEW_X, i
      double precision OBJ, X(N)
      OBJ = 0
      do i = 1, N
         OBJ = OBJ + 0.25d0*(X(i)-1d0)**4
      enddo
      return
      end

C =============================================================================
C                Computation of gradient of objective function
C =============================================================================
      subroutine EVAL_GRAD(N, X, NEW_X, GRAD)
      implicit none
      integer N, NEW_X, i
      double precision GRAD(N), X(N)
      do i = 1, N
         GRAD(i) = (X(i)-1d0)**3
      enddo
      return
      end

C =============================================================================
C                     Computation of equality constraints
C =============================================================================
      subroutine EVAL_CON(N, M, X, NEW_X, C)
      implicit none
      integer N, NEW_X, M, i
      double precision C(M), X(N)
      C(1) = 4d0*X(1) + 2d0*X(2)
      C(2) = 2d0*X(1) + 1d0*X(3)
      do i = 3, M
         C(i) = 2d0*X(1) + 0.5d0*X(i+1)
      enddo
      return
      end

C =============================================================================
C                Computation of Jacobian of equality constraints
C =============================================================================

      subroutine EVAL_JAC(N, M, X, NEW_X, NZ, A)
      integer N, NEW_X, M, NZ
      double precision X(N), A(NZ)
      integer IROW(NZ), JCOL(NZ), I, conidx

      conidx = 0
      do I = 3, M
        if( conidx.eq.0 ) then
        
        endif
          IROW(nnzit) = I
          JCOL(nnzit) = 1
          nnzit = nnzit + 1
          IROW(nnzit) = I
          JCOL(nnzit) = I + 1
          nnzit = nnzit + 1
          conidx = conidx + 1
      enddo
        
      if( TASK.eq.0 ) then
C     // constraint 1 body --->  4*x_1 + 2*x_2 == 10
        IROW(nnzit) = 1
        JCOL(nnzit) = 1
        nnzit = nnzit + 1
        IROW(nnzit) = 1
        JCOL(nnzit) = 2
        nnzit = nnzit + 1
C     // constraint 2 body ---> 2*x_1 + x_3
        IROW(nnzit) = 2
        JCOL(nnzit) = 1
        nnzit = nnzit + 1
        IROW(nnzit) = 2
        JCOL(nnzit) = 3
        nnzit = nnzit + 1
C     // constraint 3 body ---> 2*x_1 + 0.5*x_i, for i>=4
        conidx = 3
        do I = 3, M
          IROW(nnzit) = I
          JCOL(nnzit) = 1
          nnzit = nnzit + 1
          IROW(nnzit) = I
          JCOL(nnzit) = I + 1
          nnzit = nnzit + 1
          conidx = conidx + 1
        enddo
      else
C     // constraint 1 body --->  4*x_1 + 2*x_2 == 10
        A(nnzit) = 4d0
        nnzit = nnzit + 1
        A(nnzit) = 2d0
        nnzit = nnzit + 1
C     // constraint 2 body ---> 2*x_1 + x_3
        A(nnzit) = 2d0
        nnzit = nnzit + 1
        A(nnzit) = 1d0
        nnzit = nnzit + 1
C     // constraint 3 body ---> 2*x_1 + 0.5*x_i, for i>=4
        do I = 3, M
          A(nnzit) = 2d0
          nnzit = nnzit + 1
          A(nnzit) = 0.5d0
          nnzit = nnzit + 1
        enddo
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
      if( TASK.eq.0 ) then
        do I = 1, N
          IROW(I) = I
          JCOL(I) = I
        enddo
      else
         do i = 1, N
            HESS(i) = OBJFACT * 3d0 * (X(I)-1.0d0)**2
         enddo
      endif
      return
      end

