import random
import numpy as np

#################################### DO NOT CHANGE !!!!! ##################################################

# mass specs
_m_c = 1.53
_m_r = _m_l = 0.22

# inertia specs
_J_c = 0.0049

# body-rotor bridge dimension
_d_br = 0.38
_h_br = 0.05

# body dimensions
_l_b = 0.2
_w_b = 0.2

# rotor dimensions
_w_rm = _w_lm = 0.25
_h_rm = _h_lm = _w_b

# target dimensions
_l_t = 0.2
_w_t = 0.2

# geometric and initial specs
_dr_mat_i = np.array([[-_d_br + _w_lm/2, _h_br/2], [-_l_b/2, _h_br/2], [-_l_b/2, _w_b/2], [_l_b/2, _w_b/2], 
          [_l_b/2, _h_br/2], [_d_br - _w_lm/2, _h_br/2], [_d_br - _w_lm/2, _w_b/2], [_d_br + _w_lm/2, _w_b/2], [_d_br + _w_lm/2, -_w_b/2], [_d_br - _w_lm/2, -_w_b/2], [_d_br - _w_lm/2, -_h_br/2], [_l_b/2, -_h_br/2], [_l_b/2, -_w_b/2], [-_l_b/2, -_w_b/2], 
          [-_l_b/2, -_h_br/2], [-_d_br + _w_lm/2, -_h_br/2],[-_d_br + _w_lm/2, -_w_b/2], [-_d_br - _w_lm/2, -_w_b/2], [-_d_br - _w_lm/2, _w_b/2],  [-_d_br + _w_lm/2, _w_b/2], [-_d_br + _w_lm/2, _h_br/2]])

_l_init = np.array([-_d_br, _w_b/2, 0])
_r_init = np.array([_d_br, _w_b/2, 0])

def _u_c(t):
    """returns with coeffecients for Tau"""
    return np.array([-np.sin(t)/(_m_c + _m_l + _m_r), np.cos(t)/(_m_c + _m_l + _m_r), _d_br/(_J_c + (_m_l + _m_r)*(_d_br**2))])

# damping specs
_m_u = 0.121

#################################### DO NOT CHANGE !!!!! ##################################################


# initial conditions
z_i = -1
h_i = _w_b/2
theta_i = 0
target_i = -_l_t/2
zv_i = 0
hv_i = 0
thetav_i = 0
targetv_i = 0

# ground and sky dimension
l_g = 4
l_s = 4

#simulation conditions
t_start = 0
t_end = 75
t_step = 0.08
t_plot = 0.20
t_catch_up = 4
g = 9.8

########################################## PID params ONLY !!! ##############################################

# kp, kd, kdc specs
kd_z = -0.0173
kp_z = -0.003994

kp_h = 0.149
kd_h = 0.7660

kd_theta = 0.2661
kp_theta = 0.51755
kdc_theta = 1.932

########################################## PID params ONLY !!! ##############################################

# max_values:
max_force = 20
max_torque = 20

# saturating motors
def saturate_motors(a):
    """saturates left and right motor forces"""
    if a > 10:
        a = 10
    elif a < 0:
        a = 0
    else:
        pass
    return a

def uncertainity(a, u_param = 0.2):
    for i in range(len(a)):
        a[i] = a[i]*(1 + u_param*random.randint(0, 10000)/10000)
    return a
