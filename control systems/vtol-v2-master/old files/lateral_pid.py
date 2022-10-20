import numpy as np
import specs as p
from pid import PIDControl

class lateralController():
    """A PD controller for controlling the lateral displacement"""
    
    def __init__(self) -> None:
        """define manual kp, ki, kd"""
        self.z_control = PIDControl(p._o_kp_lat, p._o_ki_lat, p._o_kd_lat, p.max_theta, 0.05, p.t_step)
        self.theta_control = PIDControl(p._o_kp_theta, 0.00, p._o_kd_theta, p.max_torque, 0.05, p.t_step)

    def update(self, z_r, z_data, theta_data):
        """returns the net torque required, according to reference z"""    
        
        z = z_data[0]
        z_1 = z_data[1]
        theta = theta_data[0]
        theta_1 = theta_data[1]

        # the reference angle for theta comes from
        # the outer loop PID control
        theta_r = self.z_control.PID(z_r, z, flag=False)

        # low pass filter the outer loop to cancel
        # # left-half plane zero and DC-gain
        # theta_r = self.filter.update(theta_r)

        # the force applied to the cart comes from
        # the inner loop PD control
        F = self.theta_control.PD(theta_r, theta, flag=False)
        
        return F

        # # equilibrium
        # tau_e = 0.00
        
        # # PD controlled, outer loop 
        # theta_r = self.kp*(z_r-z) - self.kd*(z_1)

        # # PD controlled for inner loop
        # tau_tilde = self.ikp*(theta_r-theta) - self.ikd*(theta_1)
        
        # # net tau
        # net_tau = tau_tilde + tau_e

        # # saturate
        # net_tau = self.saturate(net_tau)
        
        # return net_tau

    def saturate(self, u):
        """saturates to limit physics"""
        return u