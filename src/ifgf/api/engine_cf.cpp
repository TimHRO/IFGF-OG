// FacePlan instantiations for the CF operator of the library API (one translation unit per kernel).
#include "engine.hpp"

#include <ifgf/kernels/helmholtz.hpp>

namespace ifgf::api_detail {

std::unique_ptr<Engine> makeCF(const Setup& s) { return makeEngine<kernels::HelmholtzCF>(s); }

}  // namespace ifgf::api_detail
