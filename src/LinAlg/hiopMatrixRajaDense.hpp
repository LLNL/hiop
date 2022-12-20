#pragma once
#include <cstddef>
#include <cstdio>
#include <string>

#include "hiopMatrixDenseRaja.hpp"

namespace hiop
{

  using hiopMatrixRajaDense = hiopMatrixDenseRaja<MemBackendUmpire, ExecPolicyRajaCuda>;

} // namespace hiop
