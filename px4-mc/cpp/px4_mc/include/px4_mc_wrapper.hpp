#pragma once

#include <cstdint>

extern "C" {

struct Px4McOpaque;

struct Px4McInput {
    float pos[3];
    float vel[3];
    float att_q[4]; // w, x, y, z
    float rates[3];
    float target_pos[3];
    float target_vel[3];
};

struct Px4McOutput {
    float torque[3];
    float thrust;
};

Px4McOpaque* px4_mc_create(float dt, float mass);
void px4_mc_destroy(Px4McOpaque* ptr);
void px4_mc_update(Px4McOpaque* ptr, const Px4McInput* in, Px4McOutput* out);

// Parameter setters
void px4_mc_set_att_gains(Px4McOpaque* ptr, const float p[3], float yaw_weight);
void px4_mc_set_att_rate_limit(Px4McOpaque* ptr, const float rate_limit[3]);
void px4_mc_set_pos_gains(Px4McOpaque* ptr, const float p[3]);
void px4_mc_set_vel_gains(Px4McOpaque* ptr, const float p[3], const float i[3], const float d[3]);
void px4_mc_set_vel_limits(Px4McOpaque* ptr, float vel_horizontal, float vel_up, float vel_down);
void px4_mc_set_thrust_limits(Px4McOpaque* ptr, float min_thr, float max_thr);
void px4_mc_set_thr_xy_margin(Px4McOpaque* ptr, float margin);
void px4_mc_set_tilt_limit(Px4McOpaque* ptr, float tilt_rad);
void px4_mc_set_hover_thrust(Px4McOpaque* ptr, float hover);
void px4_mc_set_rate_p(Px4McOpaque* ptr, const float p[3]);
void px4_mc_set_dt(Px4McOpaque* ptr, float dt);
void px4_mc_set_mass(Px4McOpaque* ptr, float mass);

}
