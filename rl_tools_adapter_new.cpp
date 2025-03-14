#define RL_TOOLS_DISABLE_TEST
#include "rl_tools_adapter_new.h"
#ifdef RL_TOOLS_ENABLE_DEBUGGING_POOL
char rl_tools_debugging_pool_names[RL_TOOLS_DEBUGGING_POOL_NUMBER][RL_TOOLS_DEBUGGING_POOL_NAME_LENGTH];
float rl_tools_debugging_pool[RL_TOOLS_DEBUGGING_POOL_NUMBER][RL_TOOLS_DEBUGGING_POOL_SIZE];
uint64_t rl_tools_debugging_pool_indices[RL_TOOLS_DEBUGGING_POOL_NUMBER];
uint64_t rl_tools_debugging_pool_index = 0;
bool rl_tools_debugging_pool_locked = false;
bool rl_tools_debugging_pool_updated = false;
#endif

uint64_t portable_strlen(const char* str) {
    const char* ptr = str;
    while (*ptr != '\0') {
        ptr++;
    }
    return ptr - str;
}

char *portable_strcpy(char *dest, const char *src) {
    char *original_dest = dest;
    while ((*dest++ = *src++) != '\0');
    return original_dest;
}

static void reset_debuging_pool(){
    if(!rl_tools_debugging_pool_locked){
        rl_tools_debugging_pool_index = 0;
        rl_tools_debugging_pool_updated = true;
    }
}
static void add_to_debuging_pool(const char* name, const float* values, uint64_t num){
    if(!rl_tools_debugging_pool_locked && rl_tools_debugging_pool_index < RL_TOOLS_DEBUGGING_POOL_NUMBER){
        uint32_t size = (num < RL_TOOLS_DEBUGGING_POOL_SIZE ? num : RL_TOOLS_DEBUGGING_POOL_SIZE);
        for(uint64_t i = 0; i < size; i++){
            rl_tools_debugging_pool[rl_tools_debugging_pool_index][i] = values[i];
        }
        if(portable_strlen(name) < RL_TOOLS_DEBUGGING_POOL_NAME_LENGTH){
            portable_strcpy(rl_tools_debugging_pool_names[rl_tools_debugging_pool_index], name);
        } else {
            portable_strcpy(rl_tools_debugging_pool_names[rl_tools_debugging_pool_index], "name too long");
        }
        rl_tools_debugging_pool_indices[rl_tools_debugging_pool_index] = size;
        rl_tools_debugging_pool_index++;
        rl_tools_debugging_pool_updated = true;
    }
}

#ifndef RL_TOOLS_WASM
#include <rl_tools/operations/arm.h>
#else
#include <rl_tools/operations/wasm32.h>
#endif


#include <rl_tools/nn/layers/standardize/operations_generic.h>
#ifndef RL_TOOLS_WASM
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
// #include <rl_tools/nn/layers/dense/operations_generic.h>
#else
#include <rl_tools/nn/layers/dense/operations_generic.h>
#endif
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include "data/actor.h"

#ifdef RL_TOOLS_ENABLE_INFORMATIVE_STATUS_MESSAGES
#include <cstdio>
#endif

namespace rlt = rl_tools;

#ifndef RL_TOOLS_WASM
using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
#else
using DEVICE = rlt::devices::DefaultWASM32;
#endif

using TI = typename DEVICE::index_t;
static constexpr TI TEST_BATCH_SIZE = rlt::checkpoint::example::input::SHAPE::template GET<1>;
using ACTOR_TYPE_ORIGINAL = rlt::checkpoint::actor::TYPE;
using ACTOR_TYPE_TEST = rlt::checkpoint::actor::TYPE::template CHANGE_BATCH_SIZE<TI, TEST_BATCH_SIZE>;
using ACTOR_TYPE = ACTOR_TYPE_ORIGINAL::template CHANGE_BATCH_SIZE<TI, 1>;
using T = typename ACTOR_TYPE::SPEC::T;
constexpr TI ACTION_HISTORY_LENGTH = 16; //rlt::checkpoint::environment::ACTION_HISTORY_LENGTH
constexpr TI CONTROL_INTERVAL_US_ORIGINAL = 1000 * 10; // Training is 100hz
constexpr TI CONTROL_INTERVAL_US = 1000 * 2; // Inference is at 500hz
static constexpr TI INPUT_DIM = rlt::get_last(ACTOR_TYPE::INPUT_SHAPE{});
static constexpr TI OUTPUT_DIM = rlt::get_last(ACTOR_TYPE::OUTPUT_SHAPE{});
static_assert(OUTPUT_DIM == 4);
static_assert(INPUT_DIM == (18 + ACTION_HISTORY_LENGTH * OUTPUT_DIM));
constexpr TI TIMING_STATS_NUM_STEPS = 100;

struct State{
    float position[3];
    float orientation[4]; // Quaternion: w, x, y, z
    float linear_velocity[3];
    float angular_velocity[3];
    T action_history[ACTION_HISTORY_LENGTH][OUTPUT_DIM];
    uint64_t last_observation_timestamp, last_control_timestamp, last_control_timestamp_original; // last_control_timestamp runs at the higher rate, while last_control_timestamp_original runs at the original control rate of the simulation
    bool last_observation_timestamp_set, last_control_timestamp_set, last_control_timestamp_original_set;
    typename ACTOR_TYPE::State<false> policy_state;
    uint64_t control_dt[TIMING_STATS_NUM_STEPS];
    uint64_t control_dt_index = 0;
    uint64_t control_original_dt[TIMING_STATS_NUM_STEPS];
    uint64_t control_original_dt_index = 0;
};




DEVICE device;
bool rng = false;

// Buffers
static ACTOR_TYPE_TEST::template Buffer<false> buffers_test;
static ACTOR_TYPE::template Buffer<false> buffers;
static ACTOR_TYPE::State<false> policy_state_buffer;
static rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, INPUT_DIM>, false>> input;
static rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, OUTPUT_DIM>, false>> output;
static rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::checkpoint::example::output::SHAPE, false>> output_test;

// State
State state;

template <typename OBS_SPEC>
static inline void observe(rlt::Tensor<OBS_SPEC>& observation){
    static_assert(OBS_SPEC::SHAPE::template GET<0> == 1);
    static_assert(OBS_SPEC::SHAPE::template GET<1> == 18 + OUTPUT_DIM * ACTION_HISTORY_LENGTH); // position + orientation + linear_velocity + angular_velocity + action_history
    TI base = 0;
    rlt::set(device, observation, state.position[0], 0,  base++);
    rlt::set(device, observation, state.position[1], 0,  base++);
    rlt::set(device, observation, state.position[2], 0,  base++);
    float qw = state.orientation[0];
    float qx = state.orientation[1];
    float qy = state.orientation[2];
    float qz = state.orientation[3];
    rlt::set(device, observation,   (1 - 2*qy*qy - 2*qz*qz), 0, base++);
    rlt::set(device, observation,   (    2*qx*qy - 2*qw*qz), 0, base++);
    rlt::set(device, observation,   (    2*qx*qz + 2*qw*qy), 0, base++);
    rlt::set(device, observation,   (    2*qx*qy + 2*qw*qz), 0, base++);
    rlt::set(device, observation,   (1 - 2*qx*qx - 2*qz*qz), 0, base++);
    rlt::set(device, observation,   (    2*qy*qz - 2*qw*qx), 0, base++);
    rlt::set(device, observation,   (    2*qx*qz - 2*qw*qy), 0, base++);
    rlt::set(device, observation,   (    2*qy*qz + 2*qw*qx), 0, base++);
    rlt::set(device, observation,   (1 - 2*qx*qx - 2*qy*qy), 0, base++);
    rlt::set(device, observation,  state.linear_velocity[0], 0, base++);
    rlt::set(device, observation,  state.linear_velocity[1], 0, base++);
    rlt::set(device, observation,  state.linear_velocity[2], 0, base++);
    rlt::set(device, observation, state.angular_velocity[0], 0, base++);
    rlt::set(device, observation, state.angular_velocity[1], 0, base++);
    rlt::set(device, observation, state.angular_velocity[2], 0, base++);
    for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH; step_i++){
        for(TI action_i = 0; action_i < OUTPUT_DIM; action_i++){
            rlt::set(device, observation, state.action_history[step_i][action_i], 0, base++);
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
    state.last_control_timestamp_original_set = false;
    state.control_dt_index = 0;
    state.control_original_dt_index = 0;
}
void rl_tools_init(){
    rl_tools_reset();
    rl_tools_debugging_pool_index = 0;
    rl_tools_debugging_pool_locked = false;
    rl_tools_debugging_pool_updated = false;
}


const char* rl_tools_get_checkpoint_name(){
    return (char*)rlt::checkpoint::meta::name;
}
char status_message[256] = "";
void append(const char* message, uint32_t &position){
    if(position + portable_strlen(message) < sizeof(status_message)){
        portable_strcpy(status_message + position, message);
        position += portable_strlen(message);
    }
}
char* rl_tools_get_status_message(RLtoolsStatus status){
    uint32_t position = 0;
    if(rl_tools_healthy(status)){
        append("OK", position);
        if(status & RL_TOOLS_STATUS_BIT_SOURCE_CONTROL){
            append(" CONTROL", position);
        }
        if(status & RL_TOOLS_STATUS_BIT_SOURCE_CONTROL_ORIGINAL){
            append("_ORIGINAL", position);
        }
    }
    else{
        append("PROBLEM", position);
        if(status & RL_TOOLS_STATUS_BIT_SOURCE_CONTROL){
            append(" CONTROL", position);
            if(status & RL_TOOLS_STATUS_BIT_SOURCE_CONTROL_ORIGINAL){
                append("_ORIGINAL", position);
            }
        }
        else{
            append(" UNKNOWN", position);
        }
        if(status & RL_TOOLS_STATUS_BIT_TIMESTAMP_INVALID){
            append(" TIMESTAMP_INVALID", position);
        }
        else{
            if(status & RL_TOOLS_STATUS_BIT_TIMING_ISSUE){
                append(" TIMING_ISSUE", position);
                if(status & RL_TOOLS_STATUS_BIT_TIMING_JITTER){
                    append(" JITTER", position);
                }
                if(status & RL_TOOLS_STATUS_BIT_TIMING_BIAS){
                    append(" BIAS", position);
                }
                if((status & RL_TOOLS_STATUS_BITS_MAGNITUDE) != 0){
                    T magnitude = (T)(int16_t)((status & RL_TOOLS_STATUS_BITS_MAGNITUDE) >> RL_TOOLS_STATUS_BITS_MAGNITUDE_SHIFT);
                    append(" MAGNITUDE: ", position);
#ifdef RL_TOOLS_ENABLE_INFORMATIVE_STATUS_MESSAGES
                    char buffer[50];
                    snprintf(buffer, 50, "%.2f%%", magnitude);
                    append(buffer, position);
#else
                    if(magnitude > 0){
                        append("HIGH", position);
                    }
                    else{
                        append("LOW", position);
                    }
#endif
                }
            }
            else{
                append(" UNKNOWN", position);
            }
        }

    }
    return status_message;
}

float rl_tools_test(RLtoolsAction* p_output){
#ifndef RL_TOOLS_DISABLE_TEST
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    rlt::evaluate(device, rlt::checkpoint::actor::module, rlt::checkpoint::example::input::container, output_test, buffers_test, rng, mode);
    float acc = 0;
    for(TI batch_i = 0; batch_i < TEST_BATCH_SIZE; batch_i++){
        for(TI i = 0; i < OUTPUT_DIM; i++){
            acc += rlt::math::abs(device.math, rlt::get(device, output_test, 0, batch_i, i) - rlt::get(device, rlt::checkpoint::example::output::container, 0, batch_i, i));
            if(batch_i == 0){
                p_output->action[i] = rlt::get(device, output_test, 0, batch_i, i);
            }
        }
    }
    return acc;
#else
    return 0;
#endif
}

T clip(T value, T min, T max){
    if(value < min){
        return min;
    }
    if(value > max){
        return max;
    }
    return value;
}

RLtoolsStatus timing_jitter_status(bool original){
    if((original ? state.control_original_dt_index : state.control_dt_index) < TIMING_STATS_NUM_STEPS){
        return 0;
    }
    for(TI i = 0; i < TIMING_STATS_NUM_STEPS; i++){
        auto value = original ? state.control_original_dt[i] : state.control_dt[i];
        auto expected = original ? CONTROL_INTERVAL_US_ORIGINAL : CONTROL_INTERVAL_US;
        if(value > expected * (RL_TOOLS_STATUS_TIMING_JITTER_HIGH_THRESHOLD) || value < expected * (RL_TOOLS_STATUS_TIMING_JITTER_LOW_THRESHOLD)){
            T magnitude = (value / (float)expected - 1) * 100;
            magnitude = clip(magnitude, INT16_MIN, INT16_MAX);
            uint64_t magnitude_bit = (uint16_t)((int16_t)(magnitude));
            return RL_TOOLS_STATUS_BIT_TIMING_ISSUE | RL_TOOLS_STATUS_BIT_TIMING_JITTER | (magnitude_bit << RL_TOOLS_STATUS_BITS_MAGNITUDE_SHIFT);
        }
    }
    return 0;
}

int rl_tools_healthy(RLtoolsStatus status){
    return (status & RL_TOOLS_STATUS_BITS_ISSUE) == 0;
}
float rl_tools_get_timing_bias(int original){
    if((original ? state.control_original_dt_index : state.control_dt_index) < TIMING_STATS_NUM_STEPS){
        return 0;
    }
    T value = 0;
    for(TI i = 0; i < TIMING_STATS_NUM_STEPS; i++){
        value += original ? state.control_original_dt[i] : state.control_dt[i];
    }
    value /= TIMING_STATS_NUM_STEPS;
    return value;
}

RLtoolsStatus timing_bias_status(int original){
    if((original ? state.control_original_dt_index : state.control_dt_index) < TIMING_STATS_NUM_STEPS){
        return 0;
    }
    T value = rl_tools_get_timing_bias(original);

    auto expected = original ? CONTROL_INTERVAL_US_ORIGINAL : CONTROL_INTERVAL_US;
    if(value > expected * RL_TOOLS_STATUS_TIMING_BIAS_HIGH_THRESHOLD || value < expected * RL_TOOLS_STATUS_TIMING_BIAS_LOW_THRESHOLD){
        T magnitude = (value / (float)expected - 1) * 100;
        magnitude = clip(magnitude, INT16_MIN, INT16_MAX);
        uint64_t magnitude_bit = (uint16_t)((int16_t)(magnitude));
        return RL_TOOLS_STATUS_BIT_TIMING_ISSUE | RL_TOOLS_STATUS_BIT_TIMING_BIAS | (magnitude_bit << RL_TOOLS_STATUS_BITS_MAGNITUDE_SHIFT);
    }
    return 0;
}

RLtoolsStatus rl_tools_control(uint64_t microseconds, RLtoolsObservation* observation, RLtoolsAction* action){
    bool reset = false;
    if(!state.last_observation_timestamp_set){
        state.last_observation_timestamp = microseconds;
        state.last_observation_timestamp_set = true;
    }
    if(!state.last_control_timestamp_set){
        reset = true;
        state.last_control_timestamp = microseconds;
        state.last_control_timestamp_set = true;
    }
    if(microseconds < state.last_observation_timestamp){
        state.last_observation_timestamp = microseconds;
        state.last_observation_timestamp_set = true;
        return RL_TOOLS_STATUS_BIT_TIMESTAMP_INVALID;
    }
    if(microseconds < state.last_control_timestamp){
        state.last_control_timestamp = microseconds;
        state.last_control_timestamp_set = true;
        return RL_TOOLS_STATUS_BIT_TIMESTAMP_INVALID;
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
    RLtoolsStatus status = RL_TOOLS_STATUS_OK;
    if(time_diff_control >= CONTROL_INTERVAL_US || reset){
        reset_debuging_pool();
        if(!reset){
            state.control_dt[state.control_dt_index++ % TIMING_STATS_NUM_STEPS] = time_diff_control;
        }
        state.last_control_timestamp = microseconds;
        if(!state.last_control_timestamp_original_set){
            state.last_control_timestamp_original = microseconds;
            state.last_control_timestamp_original_set = true;
        }
        uint64_t time_diff_control_original = microseconds - state.last_control_timestamp_original;
        bool real_control_step = (time_diff_control_original >= CONTROL_INTERVAL_US_ORIGINAL) || reset;
        observe(input);
        rlt::Mode<rlt::mode::Evaluation<>> mode;
        if(real_control_step){
            rlt::evaluate_step(device, rlt::checkpoint::actor::module, input, state.policy_state, output, buffers, rng, mode);
            state.last_control_timestamp_original = microseconds;
            if(!reset){
                state.control_original_dt[state.control_original_dt_index++ % TIMING_STATS_NUM_STEPS] = time_diff_control_original;
            }
            status = RL_TOOLS_STATUS_OK | RL_TOOLS_STATUS_BIT_SOURCE_CONTROL | RL_TOOLS_STATUS_BIT_SOURCE_CONTROL_ORIGINAL;
            status |= timing_jitter_status(true);
            if(rl_tools_healthy(status))
            status |= timing_bias_status(true);
        }
        else{
            policy_state_buffer = state.policy_state;
            rlt::evaluate_step(device, rlt::checkpoint::actor::module, input, policy_state_buffer, output, buffers, rng, mode);
            status = RL_TOOLS_STATUS_OK | RL_TOOLS_STATUS_BIT_SOURCE_CONTROL;
            status |= timing_jitter_status(false);
            if(rl_tools_healthy(status))
            status |= timing_bias_status(true);
        }
    }
    for(TI action_i=0; action_i < OUTPUT_DIM; action_i++){
        action->action[action_i] = rlt::get(device, output, 0, action_i);
    }
    return status;
}
