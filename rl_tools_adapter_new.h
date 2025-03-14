#ifndef RL_TOOLS_ADAPTER_NEW_H
#define RL_TOOLS_ADAPTER_NEW_H
#if defined(__cplusplus) && !defined(RL_TOOLS_WASM)
#include <cstdint>
#else
#include <stdint.h>
#endif


#ifdef RL_TOOLS_WASM
// typedef unsigned long long uint64_t;
static_assert(sizeof(uint64_t) == 8, "uint64_t must be 8 bytes");
#endif

#define RL_TOOLS_STATUS_TIMING_JITTER_LOW_THRESHOLD 0.8
#define RL_TOOLS_STATUS_TIMING_JITTER_HIGH_THRESHOLD 1.2
#define RL_TOOLS_STATUS_TIMING_BIAS_LOW_THRESHOLD 0.95
#define RL_TOOLS_STATUS_TIMING_BIAS_HIGH_THRESHOLD 1.05

#define RL_TOOLS_STATUS_OK                           (0ULL)
#define RL_TOOLS_STATUS_BIT_SOURCE_CONTROL           (1ULL << 1)
#define RL_TOOLS_STATUS_BIT_SOURCE_CONTROL_ORIGINAL  (1ULL << 2)
#define RL_TOOLS_STATUS_BITS_ISSUE_SHIFT             (8ULL)
#define RL_TOOLS_STATUS_BITS_ISSUE                   (0xfffffffULL << RL_TOOLS_STATUS_BITS_ISSUE_SHIFT)
#define RL_TOOLS_STATUS_BIT_TIMESTAMP_INVALID        (0x01ULL << (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT))
#define RL_TOOLS_STATUS_BIT_TIMING_ISSUE             (0x02ULL << (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT))
#define RL_TOOLS_STATUS_BIT_TIMING_JITTER            (1ULL << (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT + 8 + 0))
#define RL_TOOLS_STATUS_BIT_TIMING_BIAS              (1ULL << (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT + 8 + 1))
#define RL_TOOLS_STATUS_BITS_MAGNITUDE_SHIFT         (RL_TOOLS_STATUS_BITS_ISSUE_SHIFT + 8 + 1 + 16)
#define RL_TOOLS_STATUS_BITS_MAGNITUDE               (0xffffULL << RL_TOOLS_STATUS_BITS_MAGNITUDE_SHIFT)


#define RL_TOOLS_ENABLE_DEBUGGING_POOL
#ifdef RL_TOOLS_ENABLE_DEBUGGING_POOL
#define RL_TOOLS_DEBUGGING_POOL_NUMBER 10
#define RL_TOOLS_DEBUGGING_POOL_SIZE 10
#define RL_TOOLS_DEBUGGING_POOL_NAME_LENGTH 30
extern float rl_tools_debugging_pool[RL_TOOLS_DEBUGGING_POOL_NUMBER][RL_TOOLS_DEBUGGING_POOL_SIZE];
extern char rl_tools_debugging_pool_names[RL_TOOLS_DEBUGGING_POOL_NUMBER][RL_TOOLS_DEBUGGING_POOL_NAME_LENGTH];
extern uint64_t rl_tools_debugging_pool_indices[RL_TOOLS_DEBUGGING_POOL_NUMBER];
extern uint64_t rl_tools_debugging_pool_index;
extern bool rl_tools_debugging_pool_locked;
extern bool rl_tools_debugging_pool_updated;
#endif


#ifdef __cplusplus
extern "C" {
#endif
    typedef struct {
        float position[3];
        float orientation[4]; // Quaternion: w, x, y, z
        float linear_velocity[3];
        float angular_velocity[3];
        float previous_action[4];
    } RLtoolsObservation;
    typedef struct {
        float action[4];
    } RLtoolsAction;
    typedef uint64_t RLtoolsStatus;

    void rl_tools_init();
    void rl_tools_reset();
    float rl_tools_test(RLtoolsAction* action);
    // note: DON'T pass an uint32 timestamp here, which might wrap around after ~1h
    int rl_tools_healthy(RLtoolsStatus status);
    RLtoolsStatus rl_tools_control(uint64_t microseconds, RLtoolsObservation* observation, RLtoolsAction* action);
    const char* rl_tools_get_checkpoint_name();
    char* rl_tools_get_status_message(RLtoolsStatus status);
    float rl_tools_get_timing_bias(int original);
#ifdef __cplusplus
}
#endif



#endif