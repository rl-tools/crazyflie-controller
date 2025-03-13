#include "rl_tools_adapter_new.h"

#include <rl_tools/operations/arm.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
// #include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include "data/actor.h"

namespace rlt = rl_tools;

using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
using TI = typename DEVICE::index_t;
using ACTOR_TYPE_ORIGINAL = rlt::checkpoint::actor::TYPE;
using ACTOR_TYPE_TEST = rlt::checkpoint::actor::TYPE::template CHANGE_BATCH_SIZE<TI, TEST_BATCH_SIZE>;
using ACTOR_TYPE = ACTOR_TYPE_ORIGINAL::template CHANGE_BATCH_SIZE<TI, 1>;
using T = typename ACTOR_TYPE::SPEC::T;

struct State{
    float position[3];
    float orientation[4]; // Quaternion: w, x, y, z
    float linear_velocity[3];
    float angular_velocity[3];
    static T action_history[ACTION_HISTORY_LENGTH][OUTPUT_DIM];
    uint64_t last_observation_timestamp, last_control_timestamp;
    bool last_observation_timestamp_set, last_control_timestamp_set;
    typename ACTOR_TYPE::State<false> policy_state;
};

constexpr TI ACTION_HISTORY_LENGTH = 32; //rlt::checkpoint::environment::ACTION_HISTORY_LENGTH
constexpr TI CONTROL_INTERVAL_US_ORIGINAL = 1000 * 10; // Training is 100hz
constexpr TI CONTROL_INTERVAL_US = 1000 * 2; // Training is 100hz
static constexpr TI TEST_BATCH_SIZE = rlt::checkpoint::example::input::SHAPE::template GET<1>;
static constexpr TI INPUT_DIM = rlt::get_last(ACTOR_TYPE::INPUT_SHAPE{});
static constexpr TI OUTPUT_DIM = rlt::get_last(ACTOR_TYPE::OUTPUT_SHAPE{});
static_assert(OUTPUT_DIM == 4);
static_assert(INPUT_DIM == (18 + ACTION_HISTORY_LENGTH * OUTPUT_DIM));

DEVICE device;
bool rng = false;

// Buffers
static ACTOR_TYPE_TEST::template Buffer<false> buffers_test;
static ACTOR_TYPE::template Buffer<false> buffers;
static ACTOR_TYPE::State<false> policy_state_buffer;
static rlt::Matrix<rlt::matrix::Specification<T, TI, 1, INPUT_DIM, false>> input;
static rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::checkpoint::example::output::SHAPE, false>> output_test;

// State
State state;

template <typename STATE_SPEC, typename OBS_SPEC>
static inline void observe(const State& state, rlt::Matrix<OBS_SPEC>& observation){
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS == 18);
    TI base = 0;
    rlt::set(observation, 0,  base++, state.position[0]);
    rlt::set(observation, 0,  base++, state.position[1]);
    rlt::set(observation, 0,  base++, state.position[2]);
    float qw = state.orientation[0];
    float qx = state.orientation[1];
    float qy = state.orientation[2];
    float qz = state.orientation[3];
    rlt::set(observation, 0, base++, (1 - 2*qy*qy - 2*qz*qz));
    rlt::set(observation, 0, base++, (    2*qx*qy - 2*qw*qz));
    rlt::set(observation, 0, base++, (    2*qx*qz + 2*qw*qy));
    rlt::set(observation, 0, base++, (    2*qx*qy + 2*qw*qz));
    rlt::set(observation, 0, base++, (1 - 2*qx*qx - 2*qz*qz));
    rlt::set(observation, 0, base++, (    2*qy*qz - 2*qw*qx));
    rlt::set(observation, 0, base++, (    2*qx*qz - 2*qw*qy));
    rlt::set(observation, 0, base++, (    2*qy*qz + 2*qw*qx));
    rlt::set(observation, 0, base++, (1 - 2*qx*qx - 2*qy*qy));
    rlt::set(observation, 0, base++, state.linear_velocity[0]);
    rlt::set(observation, 0, base++, state.linear_velocity[1]);
    rlt::set(observation, 0, base++, state.linear_velocity[2]);
    rlt::set(observation, 0, base++, state.angular_velocity[0]);
    rlt::set(observation, 0, base++, state.angular_velocity[1]);
    rlt::set(observation, 0, base++, state.angular_velocity[2]);
    for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH; step_i++){
        for(TI action_i = 0; action_i < OUTPUT_DIM; action_i++){
            rlt::set(observation, 0, base++, action_history[step_i][action_i]);
        }
    }
}

// Main functions (possibly with side effects)
void rl_tools_reset(){
    constexpr T HOVERING_THROTTLE = 0.66;
    for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH; step_i++){
        for(TI action_i = 0; action_i < OUTPUT_DIM; action_i++){
            state.action_history[step_i][action_i] = HOVERING_THROTTLE * 2 - 1;
        }
    }
    rlt::reset(device, rlt::checkpoint::actor::module, state.policy_state, rng);
    state.last_observation_timestamp_set = false;
    state.last_control_timestamp_set = false;
}
void rl_tools_init(){
    rl_tools_reset();
}

char* rl_tools_get_checkpoint_name(){
    return (char*)rlt::checkpoint::meta::name;
}

char* rl_tools_get_status_name(RLtoolsStatus status){
    switch(status){
        case RL_TOOLS_STATUS_OK:
            return "OK";
        case RL_TOOLS_STATUS_TIMESTAMP_INVALID:
            return "Timestamp invalid";
        default:
            return "Unknown status";
    }
}

float rl_tools_test(RLtoolsAction* output){
#ifndef RL_TOOLS_DISABLE_TEST
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    rlt::evaluate(device, rlt::checkpoint::actor::module, rlt::checkpoint::example::input::container, output_test, buffers_test, rng, mode);
    float acc = 0;
    for(TI batch_i = 0; batch_i < TEST_BATCH_SIZE; batch_i++){
        for(TI i = 0; i < OUTPUT_DIM; i++){
            acc += rlt::math::abs(device.math, rlt::get(device, output_test, 0, batch_i, i) - rlt::get(device, rlt::checkpoint::example::output::container, 0, batch_i, i));
            if(batch_i == 0){
                output->action[i] = rlt::get(device, output_test, 0, batch_i, i);
            }
        }
    }
    return acc;
#else
    return 0;
#endif
}

int rl_tools_control(uint64_t microseconds, RLtoolsObservation* observation, RLtoolsAction* action){
    if(!state.last_observation_timestamp_set){
        state.last_observation_timestamp = microseconds;
        state.last_observation_timestamp_set = true;
    }
    if(!state.last_control_timestamp_set){
        state.last_control_timestamp = microseconds;
        state.last_control_timestamp_set = true;
    }
    if(microseconds <= state.last_observation_timestamp){
        state.last_observation_timestamp = microseconds;
        state.last_observation_timestamp_set = true;
        return RL_TOOLS_STATUS_TIMESTAMP_INVALID;
    }
    if(microseconds <= state.last_control_timestamp){
        state.last_control_timestamp = microseconds;
        state.last_control_timestamp_set = true;
        return RL_TOOLS_STATUS_TIMESTAMP_INVALID;
    }

    uint64_t time_diff_obs = microseconds - state.last_observation_timestamp;
    uint64_t time_diff_previous_obs = state.last_observation_timestamp - state.last_control_timestamp;
    uint64_t time_diff_control = microseconds - state.last_control_timestamp;

    if(state.last_control_timestamp >= state.last_observation_timestamp){
        for(TI i=0; i<3; i++){
            state.position[i] = observation->position[i];
            state.orientation[i] = observation->orientation[i];
            state.linear_velocity[i] = observation->linear_velocity[i];
            state.angular_velocity[i] = observation->angular_velocity[i];
        }
        state.orientation[3] = observation->orientation[3]; // z
        static_assert(ACTION_HISTORY_LENGTH >= 1);
        for(TI step_i = ACTION_HISTORY_LENGTH-1; step_i > 0; step_i--){
            for(TI action_i = 0; action_i < OUTPUT_DIM; action_i++){
                state.action_history[step_i][action_i] = state.action_history[step_i-1][action_i];
            }
        }
        for(TI action_i = 0; action_i < OUTPUT_DIM; action_i++){
            state.action_history[0][action_i] = observation->previous_action[action_i];
        }
    }
    else{
        float obs_weight = (float)time_diff_obs / (float)time_diff_control;
        float prev_obs_weight = (float)time_diff_previous_obs / (float)time_diff_control;

        for(TI i=0; i<3; i++){
            state.position[i]         = state.position[i]         * prev_obs_weight + obs_weight * observation->position[i];
            state.orientation[i]      = state.orientation[i]      * prev_obs_weight + obs_weight * observation->orientation[i];
            state.linear_velocity[i]  = state.linear_velocity[i]  * prev_obs_weight + obs_weight * observation->linear_velocity[i];
            state.angular_velocity[i] = state.angular_velocity[i] * prev_obs_weight + obs_weight * observation->angular_velocity[i];
        }
        state.orientation[3] = observation->orientation[3]; // z
        for(TI action_i = 0; action_i < OUTPUT_DIM; action_i++){
            state.action_history[0][action_i] = state.action_history[0][action_i] * prev_obs_weight + obs_weight * observation->previous_action[action_i];
        }
    }
    if(time_diff_control >= CONTROL_INTERVAL_US){
        rlt::Mode<rlt::mode::Evaluation<>> mode;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, OUTPUT_DIM>, true>> output = {action->action};
        bool real_control_step = time_diff_control >= CONTROL_INTERVAL_US_ORIGINAL;
        if(!real_control_step){
            policy_state_buffer = state.policy_state;
        }
        else{
            state.last_control_timestamp = microseconds;
        }
        auto& policy_state = real_control_step ? state.policy_state : policy_state_buffer;
        rlt::evaluate_step(device, rlt::checkpoint::actor::module, input, policy_state, output, buffers, rng, mode);
    }
    return RL_TOOLS_STATUS_OK;
}