#include "hiopVectorRaja.hpp"

#include "ExecSpace.hpp"
//#include "MemorySpaceCuda.hpp"
#include "ExecPoliciesRajaCuda.hpp"

namespace hiop
{

using hiop_raja_exec2 = ExecRajaPoliciesBackend<ExecPolicyRajaCuda>::hiop_raja_exec;
//hiop::ExecSpace<MemBackendUmpire,ExecPolicyRajaCuda>::policy_backend()::hiop_raja_exec;
//hiop::HiopGetPolicy<hiop::ExecPoliciesRajaCuda>::hiop_raja_exec;
}

#include "hiopVectorRajaImpl.hpp"

namespace hiop
{

template hiopVectorRaja<MemBackendUmpire,ExecPolicyRajaCuda>::hiopVectorRaja(const size_type& glob_n,
                                                                                 std::string mem_space /* = "HOST" */,
                                                                                 index_type* col_part /* = NULL */,
                                                                                 MPI_Comm comm /* = MPI_COMM_NULL */);

}
