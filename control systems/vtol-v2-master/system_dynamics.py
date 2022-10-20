import numpy as np
import matplotlib.pyplot as plt
import specs as p

# for length
# let x = (z, z') = (x1, x2)
# therfore x' = (z', u - (mu)z' ) = (x2, 0x1 - (mu/M)x2 + u + 0 ) = f(x,u) :: u = -(fl+fr)sint/M
# z = x1 = h(x)

# for height
# let x = (h, h') = (x1, x2)
# therfore x' = (h', u/M - g ) = (x2, 0x1 + 0x2 + u - g ) = f(x,u) :: u = (fl+fr)cost/M
# h = x1 = h(x)

# for rotation:
# let x = (t, t') = (x1, x2)
# therfore x' = (t', u/j) = (x2, 0x1 + 0x2 + u + 0 ) = f(x,u) :: u = d(fr-fl)/J
# t = x1 = h(x)

class systemDynamics:

    def __init__(self, init_state, time_step) -> None:   
        """initializes system and sets EL equation params"""
        # defining parameters for EL equations
        self.state = np.array(init_state)
        self.params = [[0, 0, 0], [-(p._m_u/(p._m_c + p._m_l + p._m_r)), 0, 0]]
        self.constants = [0, -p.g, 0]
        self.time_step = time_step

    def f(self, state, u):
        """returns derivative of x"""
        y_1 = state[1]
        y_2 = np.sum(self.params*state, axis=0) + np.array(self.constants) + u
        x_1 = np.array([y_1, y_2])
        return x_1
    
    def h(self):
        """returns y, i.e., x1"""
        return self.state[0]

    def update(self, u):
        """updates state and returns current state"""
        self.rk4_step(u)
        self.limit()
        return self.h()

    def rk4_step(self, u):
        """integrates ode with rk4 algo"""
        f1 = self.f(self.state, u)
        f2 = self.f(self.state + self.time_step/2 * f1, u)
        f3 = self.f(self.state + self.time_step/2 * f2, u)
        f4 = self.f(self.state + self.time_step * f3, u)
        self.state += self.time_step/6 * (f1 + 2*f2 + 2*f3 + f4)

    def limit(self) -> None:
        """introduces physical limitations"""
        if self.state[0][1] < p._w_b/2:
            self.state[0][1] = p._w_b/2