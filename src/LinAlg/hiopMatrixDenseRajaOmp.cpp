#include "hiopMatrixDenseRaja.hpp"

#include "MemBackendUmpireImpl.hpp"
#include "MemBackendCppImpl.hpp"
#include "ExecPoliciesRajaOmpImpl.hpp"


namespace hiop
{
using hiop_raja_exec = ExecRajaPoliciesBackend<ExecPolicyRajaOmp>::hiop_raja_exec;
using hiop_raja_reduce = ExecRajaPoliciesBackend<ExecPolicyRajaOmp>::hiop_raja_reduce;
}

#include "hiopMatrixDenseRajaImpl.hpp"

namespace hiop
{

//
//Explicit instantiations: force compilation 
//
template class hiopMatrixDenseRaja<MemBackendUmpire, ExecPolicyRajaOmp>;
template class hiopMatrixDenseRaja<MemBackendCpp, ExecPolicyRajaOmp>;
}
