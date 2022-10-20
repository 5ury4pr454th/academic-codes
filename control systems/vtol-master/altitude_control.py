import numpy as np
import specs as p

class AltitudeController():
    
    """A PD controller for controlling the altitude"""
    
    def __init__(self, saturate_at_0 = True, kp_h = p.kp_h, kd_h = p.kd_h) -> None:
        """define manual kp, kd"""

        self.kp = kp_h
        self.kd = kd_h

        self.limit = p.max_force
        self.saturate_at_0 = saturate_at_0

    def update(self, h_r, h_data):
        """returns the net force required, according to reference h"""    
        h = h_data[0]
        h_1 = h_data[1]

        # equilibrium
        theta_e = 0.00
        F_e = (p._m_c + p._m_r + p._m_l)*p.g*np.cos(theta_e)
        
        # PD controlled
        F_tilde = self.kp*(h_r-h) - self.kd*(h_1)
        
        # net force
        net_F = F_tilde + F_e

        # saturate
        net_F = self.saturate(net_F)
        
        return net_F

    def saturate(self, u):
        
        """saturates to limit physics"""
        if abs(u) > self.limit:
            u = self.limit*np.sign(u)
        if self.saturate_at_0 == True and u < 0:
            u = 0.00
        return u
