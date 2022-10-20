import numpy as np
import specs as p
import control_lib as ctr_lb


class lateralController():
    """A PD controller for controlling the lateral displacement"""
    
    def __init__(self, K, kr) -> None:
        """define manual kp, kd"""
        self.K = K
        self.kr = kr
        self.limit = p.max_torque

    def update(self, z_r, z_data, theta_data):
        """returns the net torque required, according to reference z"""    

        x_data = np.array([z_data[0], theta_data[0], z_data[1], theta_data[1]])
        x_data = np.expand_dims(x_data, axis = 0).T

        # equilibrium
        tau_e = 0.00
        
        # PD controlled, outer loop 
        tau_tilde = np.squeeze(-self.K@x_data) + self.kr*z_r
        
        # net tau
        net_tau = tau_tilde + tau_e

        # saturate
        net_tau = self.saturate(net_tau)
        
        return net_tau

    def saturate(self, u):
        """saturates to limit physics"""

        if np.abs(u) > self.limit:
            return self.limit*np.sign(u)
        else:
            return u



class altitudeController():
    """A PD controller for controlling the longitudinal displacement"""
    
    def __init__(self, K, kr) -> None:
        """define manual kp, kd"""
        self.K = K
        self.kr = kr
        self.limit = p.max_force
        
    def update(self, h_r, h_data):
        """returns the net force required, according to reference h"""    
    
        # equilibrium
        theta_e = 0.00
        force_e = (p._m_c + p._m_r + p._m_l)*p.g*np.cos(theta_e)
        
        x_data = np.expand_dims(h_data, axis = 0).T
        # PD controlled 
        force_tilde = np.squeeze(-self.K@x_data) + self.kr*h_r

        # net force
        net_force = force_tilde + force_e
        
        # saturate
        net_force = self.saturate(net_force)
        
        return net_force

    def saturate(self, u):
        """saturates to limit physics"""
        
        if np.abs(u) > self.limit:
            return self.limit * np.sign(u)
        else:
            return u