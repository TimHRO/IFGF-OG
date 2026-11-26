// FacePlan instantiations for the SL operator of the library API (one translation unit per kernel).
#include "engine.hpp"

#include <ifgf/kernels/helmholtz.hpp>

namespace ifgf::api_detail {

std::unique_ptr<Engine> makeSL(const Setup& s) { return makeEngine<kernels::HelmholtzSL>(s); }

}  // namespace ifgf::api_detail
