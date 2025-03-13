#include "rl_tools_adapter_new.h"
#include <iostream>
#include <random>
static constexpr uint OUTPUT_DIM = 4;
int main(){
    rl_tools_init();
    RLtoolsAction action;
    float test_result = rl_tools_test(&action);
    std::cout << "test: " << test_result << std::endl;
    for(uint i = 0; i < OUTPUT_DIM; i++){
        std::cout << "action[" << i << "] = " << action.action[i] << std::endl;
    }
    uint64_t timestamp = 0;
    RLtoolsObservation observation;
    observation.position[0] = 0.0f;
    observation.position[1] = 0.0f;
    observation.position[2] = 0.0f;
    observation.orientation[0] = 1.0f;
    observation.orientation[1] = 0.0f;
    observation.orientation[2] = 0.0f;
    observation.orientation[3] = 0.0f;
    observation.linear_velocity[0] = 0.0f;
    observation.linear_velocity[1] = 0.0f;
    observation.linear_velocity[2] = 0.0f;
    observation.angular_velocity[0] = 0.0f;
    observation.angular_velocity[1] = 0.0f;
    observation.angular_velocity[2] = 0.0f;
    for(uint j = 0; j < OUTPUT_DIM; j++){
        observation.previous_action[j] = 0.0f;
    }
    std::default_random_engine rng;
    RLtoolsStatus status;
    for(uint64_t timestamp=0; timestamp < 10000000;){
        status = rl_tools_control(timestamp, &observation, &action);
        if(status != RL_TOOLS_STATUS_OK){

        }
        std::cout << timestamp << " status: " << rl_tools_get_status_message(status) << std::endl;
        // for(uint i = 0; i < OUTPUT_DIM; i++){
        //     std::cout << "action[" << i << "] = " << action.action[i] << std::endl;
        // }
        timestamp += 2500; //std::uniform_int_distribution<uint>(500, 5000)(rng);
    }
    return 0;
}