# PX4 source list used by this binding.
# Uses the PX4 checkout pointed to by PX4_SRC_DIR (passed from env).

set(PX4_MC_PX4_SOURCES
    # Attitude control
    ${PX4_SRC_DIR}/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp

    # Position control core + math helpers
    ${PX4_SRC_DIR}/src/modules/mc_pos_control/PositionControl/PositionControl.cpp
    ${PX4_SRC_DIR}/src/modules/mc_pos_control/PositionControl/ControlMath.cpp

    # Rate control core library
    ${PX4_SRC_DIR}/src/lib/rate_control/rate_control.cpp
)
