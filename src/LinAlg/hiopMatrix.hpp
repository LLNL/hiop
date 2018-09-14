// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read “Additional BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list 
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the disclaimer (as noted below) in the documentation and/or 
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written 
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department 
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under 
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC 
// nor any of their employees, makes any warranty, express or implied, or assumes any 
// liability or responsibility for the accuracy, completeness, or usefulness of any 
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by 
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or 
// imply its endorsement, recommendation, or favoring by the United States Government or 
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed 
// herein do not necessarily state or reflect those of the United States Government or 
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or 
// product endorsement purposes.

#ifndef HIOP_MATRIX
#define HIOP_MATRIX

#include "hiop_defs.hpp"

#ifdef HIOP_USE_MPI
#include "mpi.h"
#else 

#ifndef MPI_COMM
#define MPI_Comm int
#endif
#ifndef MPI_COMM_SELF
#define MPI_COMM_SELF 0
#endif
#include <cstddef>

#endif 

#include <cstdio>

namespace hiop
{

class hiopVector;
class hiopVectorPar;

class hiopMatrix
{
public:
  virtual ~hiopMatrix() {};
  virtual void setToZero()=0;
  virtual void setToConstant(double c)=0;

  /** y = beta * y + alpha * this * x */
  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x ) const = 0;

  /** y = beta * y + alpha * this^T * x */
  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha,  const hiopVector& x ) const = 0;
  /* W = beta*W + alpha*this*X */  
  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const = 0;

  /* W = beta*W + alpha*this^T*X */
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const =0;

  /* W = beta*W + alpha*this*X^T */
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const =0;

  virtual void addDiagonal(const hiopVector& d_) = 0;
  virtual void addDiagonal(const double& value) = 0;
  virtual void addSubDiagonal(long long start, const hiopVector& d_) = 0;

  /* this += alpha*X */
  virtual void addMatrix(double alpah, const hiopMatrix& X) = 0;

  virtual double max_abs_value() = 0;

  /* return false is any of the entry is a nan, inf, or denormalized */
  virtual bool isfinite() const = 0;
  
  /* call with -1 to print all rows, all columns, or on all ranks; otherwise will
  *  will print the first rows and/or columns on the specified rank.
  */
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const = 0;
  /* number of rows */
  virtual long long m() const = 0;
  /* number of columns */
  virtual long long n() const = 0;
#ifdef HIOP_DEEPCHECKS
  /* check symmetry */
  virtual bool assertSymmetry(double tol=1e-16) const = 0;
#endif
private:
};

/** Dense matrix stored row-wise and distributed column-wise 
 */
class hiopMatrixDense : public hiopMatrix
{
public:
  hiopMatrixDense(const long long& m, const long long& glob_n, long long* col_part=NULL, MPI_Comm comm=MPI_COMM_SELF, const long long& m_max_alloc=-1);
  virtual ~hiopMatrixDense();

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void copyFrom(const hiopMatrixDense& dm);
  virtual void copyFrom(const double* buffer);

  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const;
  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const;

  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;
  virtual void timesMat_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  //to be used only locally
  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;
  //to be used only locally
  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;
  virtual void timesMatTrans_local(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void addDiagonal(const hiopVector& d_);
  virtual void addDiagonal(const double& value);
  virtual void addSubDiagonal(long long start, const hiopVector& d_);

  virtual void addMatrix(double alpah, const hiopMatrix& X);
  virtual double max_abs_value();

  virtual bool isfinite() const;
  
  //virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const;

  virtual hiopMatrixDense* alloc_clone() const;
  virtual hiopMatrixDense* new_copy() const;

  void appendRow(const hiopVectorPar& row);
  /*copy 'num_rows' rows from 'src' in this starting at 'row_dest' */
  void copyRowsFrom(const hiopMatrixDense& src, int num_rows, int row_dest);
  /* copies 'src' into this as a block starting at (i_block_start,j_block_start) */
  /*copyMatrixAsBlock */
  void copyBlockFromMatrix(const long i_block_start, const long j_block_start,
			   const hiopMatrixDense& src);
  
  /* overwrites 'this' with 'src''s block starting at (i_src_block_start,j_src_block_start) and dimensions of this */
  void copyFromMatrixBlock(const hiopMatrixDense& src, const int i_src_block_start, const int j_src_block_start);
  /*  shift<0 -> up; shift>0 -> down  */
  void shiftRows(long long shift);
  void replaceRow(long long row, const hiopVectorPar& vec);
  /* copies row 'irow' in the vector 'row_vec' (sizes should match) */
  void getRow(long long irow, hiopVector& row_vec);
#ifdef HIOP_DEEPCHECKS
  void overwriteUpperTriangleWithLower();
  void overwriteLowerTriangleWithUpper();
#endif
  inline long long get_local_size_n() const { return n_local; }
  inline long long get_local_size_m() const { return m_local; }

  inline double** local_data() const {return M; }
  inline double*  local_buffer() const {return M[0]; }
  //do not use this unless you sure you know what you're doing
  inline double** get_M() { return M; }

  virtual long long m() const {return m_local;}
  virtual long long n() const {return n_global;}
#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const;
#endif
private:
  double** M; //local storage
  long long n_global; //total / global number of columns
  int m_local, n_local; //local number of rows and cols, respectively
  long long glob_jl, glob_ju;
  MPI_Comm comm;
  
  //this is very private do not touch :)
  long long max_rows;
private:
  hiopMatrixDense() {};
  /** copy constructor, for internal/private use only (it doesn't copy the values) */
  hiopMatrixDense(const hiopMatrixDense&);
};

}
#endif
