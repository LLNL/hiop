
/*
 * This header defines a couple of types used by HiOp. 
 * 
 * This header is shared by both C and C++ interfaces of HiOp and this is
 * the only place where these types should/need to be modified.
 */

//
// int types 
//
typedef int hiop_index_type;
typedef int hiop_size_type;
#ifdef HIOP_USE_MPI
#include "mpi.h"
// MPI_Datatype corresponding to the above types should be specified below
#define MPI_HIOP_INDEX_TYPE MPI_INT
#define MPI_HIOP_SIZE_TYPE MPI_INT
#endif

//
// Same types as above but in C++
//
#ifdef __cplusplus
namespace hiop
{
  using index_type = hiop_index_type;
  using size_type  = hiop_size_type;
}
#endif
