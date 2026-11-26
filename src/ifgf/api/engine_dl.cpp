// FacePlan instantiations for the DL operator of the library API (one translation unit per kernel).
#include "engine.hpp"

#include <ifgf/kernels/helmholtz.hpp>

namespace ifgf::api_detail {

std::unique_ptr<Engine> makeDL(const Setup& s) { return makeEngine<kernels::HelmholtzDL>(s); }

}  // namespace ifgf::api_detail
