#pragma once

#include "hiopVectorPar.hpp"
#include "hiopMatrixDense.hpp"
#include "hiopMatrixDenseRowMajor.hpp"

namespace hiop
{
/**
 * TODO: better document the use cases, responsibilities, pre/post conditions
 * of the factory method
 */
inline hiopMatrixDense* getMatrixDenseInstance(const long long& m, const long long& glob_n,
  long long* col_part = NULL, MPI_Comm comm = MPI_COMM_SELF, const long long& m_max_alloc = -1)
{
  return new hiopMatrixDenseRowMajor(m, glob_n, col_part, comm, m_max_alloc);
}

inline hiopVector* getVectorInstance(
  const long long& glob_n, long long* col_part = NULL, MPI_Comm comm = MPI_COMM_NULL)
{
  return new hiopVectorPar(glob_n, col_part, comm);
}

}   // namespace hiop
