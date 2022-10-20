import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import specs as p

plt.ion()
class vtolAnimation:

    def __init__(self) -> None:
        """creates ground and sky limits, initializes plots"""
        self.first = self.first_t = True
        self.fig, self.ax = plt.subplots()
        self.handle = dict()

        self.ax.plot([-100*p.l_g, 100*p.l_g], [0.00,0.00], color = 'black')      
        self.ax.set_xlim(-p.l_g, p.l_g)
        self.ax.set_ylim(-0.5, p.l_s)

    
    def update(self, z, h, theta, target = 0) -> None:
        """updates all parameters"""

        # updating the drone:

        # under rotation of angle q:
        # (a,b) -> (acosq - bsinq, asinq + bcosq)
        
        # the rotation matrix involved
        # [[cosq, -sinq]
        #  [sinq, cosq]]

        self.ax.set_xlim(z-p.l_g/2, z+p.l_g/2)
        self.ax.set_ylim(h-p.l_s/4, h + 3*p.l_s/4)

        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                 [np.sin(theta), np.cos(theta)]])
        stats = np.matmul(rot_mat, p._dr_mat_i.T).T

        # translation by z and h
        stats = np.add(stats, np.array([z, h])) 

        # update animation
        self.draw_drone(stats)
        if target != 0:
            self.draw_target(target)

    def draw_drone(self, stats) -> None:
        """draws drone, updates if already exists"""
        if self.first == True:
            drone = mpatches.Polygon(np.add(p._dr_mat_i, [p.z_i, p.h_i]), closed = True)
            self.handle['drone_body'] = drone
            self.ax.add_patch(drone)
            self.first = False
        else:
            self.handle['drone_body'].set_xy(stats)

    def draw_target(self, x) -> None:
        """draws target, updates if already exists"""
        if self.first_t == True:
            target_body = mpatches.Rectangle([p.target_i, 0], p._l_t, p._w_t, color = 'red', alpha = 0.9)
            self.handle['target'] = target_body
            self.ax.add_patch(self.handle['target'])
            self.first_t = False
        else:
            self.handle['target'].set_x(x - p._l_t/2)