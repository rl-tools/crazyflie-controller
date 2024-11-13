#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include "data/actor.h"

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using TI = DEVICE::index_t;
using TYPE = rl_tools::checkpoint::actor::TYPE;
using T = typename TYPE::T;

#include <iostream>

int main(){
    TYPE::template Buffer<false> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rl_tools::checkpoint::example::output::SHAPE, false>> output;

    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    rlt::evaluate(device, rl_tools::checkpoint::actor::module, rl_tools::checkpoint::example::input::container, output, buffer, rng, mode);
    rlt::log(device, device.logger, "Result: ");
    rlt::print(device, output);
    rlt::log(device, device.logger, "Expected: ");
    rlt::print(device, rl_tools::checkpoint::example::output::container);
    T abs_diff = rlt::abs_diff(device, output, rl_tools::checkpoint::example::output::container);
    rlt::log(device, device.logger, "Difference: ", abs_diff);
    return abs_diff < 1e-6 ? 0 : abs_diff * 1e6;
}


// g++ checkpoint_test.cpp -I external/rl_tools/include/ && ./a.out; echo $?