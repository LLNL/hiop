#include "hiopVectorRaja.hpp"

#include "MemBackendUmpireImpl.hpp"
#include "ExecPoliciesRajaCudaImpl.hpp"


namespace hiop
{
using hiop_raja_exec2 = ExecRajaPoliciesBackend<ExecPolicyRajaCuda>::hiop_raja_exec;
using hiop_raja_reduce = ExecRajaPoliciesBackend<ExecPolicyRajaCuda>::hiop_raja_reduce;
}

#include "hiopVectorRajaImpl.hpp"

namespace hiop
{

template hiopVectorRaja<MemBackendUmpire,ExecPolicyRajaCuda>::hiopVectorRaja(const size_type& glob_n,
                                                                                 std::string mem_space /* = "HOST" */,
                                                                                 index_type* col_part /* = NULL */,
                                                                                 MPI_Comm comm /* = MPI_COMM_NULL */);

}
