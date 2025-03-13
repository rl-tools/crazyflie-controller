#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif


struct RLtoolsObservation{
    float position[3];
    float orientation[4]; // Quaternion: w, x, y, z
    float linear_velocity[3];
    float angular_velocity[3];
    float previous_action[4];
};
struct RLtoolsAction{
    float action[4];
};

enum RLtoolsStatus{
    RL_TOOLS_STATUS_OK = 0,
    RL_TOOLS_STATUS_TIMESTAMP_INVALID = 1
};

#ifdef __cplusplus
extern "C" {
#endif
    void rl_tools_init();
    void rl_tools_reset();
    float rl_tools_test(RLtoolsAction* action);
    // note: DON'T pass an uint32 timestamp here, which might wrap around after ~1h
    int rl_tools_control(uint64_t microseconds, RLtoolsObservation* observation, RLtoolsAction* action);
    char* rl_tools_get_checkpoint_name();
    char* rl_tools_get_status_name(RLtoolsStatus status);
#ifdef __cplusplus
}
#endif


