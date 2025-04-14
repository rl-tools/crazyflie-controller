#include <rl_tools/operations/arm.h>

#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include "data/actor_recurrent.h"

namespace rlt = rl_tools;

using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;


using TI = typename DEVICE::index_t;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
static constexpr TI TEST_SEQUENCE_LENGTH = rlt::checkpoint::example::input::SHAPE::template GET<0>;
static constexpr TI TEST_BATCH_SIZE = rlt::checkpoint::example::input::SHAPE::template GET<1>;
static constexpr TI TEST_SEQUENCE_LENGTH_ACTUAL = 5;
static constexpr TI TEST_BATCH_SIZE_ACTUAL = 2;
static_assert(TEST_BATCH_SIZE_ACTUAL <= TEST_BATCH_SIZE);
static_assert(TEST_SEQUENCE_LENGTH_ACTUAL <= TEST_SEQUENCE_LENGTH);
using ACTOR_TYPE_ORIGINAL = rlt::checkpoint::actor::TYPE;
using ACTOR_TYPE_TEST = rlt::checkpoint::actor::TYPE::template CHANGE_BATCH_SIZE<TI, TEST_BATCH_SIZE_ACTUAL>::template CHANGE_SEQUENCE_LENGTH<TI, TEST_SEQUENCE_LENGTH_ACTUAL>;
using ACTOR_TYPE = ACTOR_TYPE_ORIGINAL::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_SEQUENCE_LENGTH<TI, 1>;
auto& rl_tools_inference_applications_l2f_policy = rlt::checkpoint::actor::module;

// #define RL_TOOLS_DISABLE_TEST
#include <rl_tools/inference/applications/l2f/c_backend.h>

