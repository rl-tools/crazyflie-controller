#include "rl_tools_adapter_new.h"
#include <emscripten/bind.h>

RLtoolsAction rl_tools_test_by_value(){
    RLtoolsAction action;
    rl_tools_test(&action);
    return action;
}

RLtoolsStatus rl_tools_control_by_value(uint64_t microseconds, RLtoolsObservation observation, RLtoolsAction action){
    return rl_tools_control(microseconds, &observation, &action);
}

std::string rl_tools_get_checkpoint_name_by_value(){
    return std::string(rl_tools_get_checkpoint_name());
}
std::string rl_tools_get_status_message_by_value(RLtoolsStatus status){
    return std::string(rl_tools_get_status_message(status));
}

using namespace emscripten;

EMSCRIPTEN_BINDINGS(rl_tools_adapter){
    value_object<RLtoolsObservation>("RLtoolsObservation")
        .field("position", &RLtoolsObservation::position)
        .field("orientation", &RLtoolsObservation::orientation)
        .field("linear_velocity", &RLtoolsObservation::linear_velocity)
        .field("angular_velocity", &RLtoolsObservation::angular_velocity)
        .field("previous_action", &RLtoolsObservation::previous_action)
    ;
    value_object<RLtoolsAction>("RLtoolsAction")
        .field("action", &RLtoolsAction::action)
    ;
    function("rl_tools_init", &rl_tools_init);
    function("rl_tools_reset", &rl_tools_reset);
    function("rl_tools_test", &rl_tools_test_by_value);
    function("rl_tools_healthy", &rl_tools_healthy);
    function("rl_tools_control", &rl_tools_control_by_value);
    function("rl_tools_get_checkpoint_name", &rl_tools_get_checkpoint_name_by_value);
    function("rl_tools_get_status_message", &rl_tools_get_status_message_by_value);
}