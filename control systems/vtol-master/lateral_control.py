import numpy as np
import specs as p

class lateralController():
    """A PD controller for controlling the lateral displacement"""
    
    def __init__(self, kp_z = p.kp_z, kd_z = p.kd_z, kdc = p.kdc_theta) -> None:
        """define manual kp, kd"""
        self.kp = kp_z
        self.kd = kd_z
        self.ikp = p.kp_theta
        self.ikd = p.kd_theta
        self.kdc = kdc
        self.limit = p.max_torque

    def update(self, z_r, z_data, theta_data):
        """returns the net torque required, according to reference z"""    
        z = z_data[0]
        z_1 = z_data[1]
        theta = theta_data[0]
        theta_1 = theta_data[1]


        # equilibrium
        tau_e = 0.00
        
        # PD controlled, outer loop 
        theta_r = self.kp*(z_r-z) - self.kd*(z_1)

        # PD controlled for inner loop
        tau_tilde = self.ikp*(theta_r-theta) - self.ikd*(theta_1)
        
        # net tau
        net_tau = tau_tilde + tau_e

        # saturate
        net_tau = self.saturate(net_tau)
        
        return net_tau

    def saturate(self, u):
        """saturates to limit physics"""
        return u