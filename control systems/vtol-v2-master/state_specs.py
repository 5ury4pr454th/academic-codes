import numpy as np

# for latitude
A_lat = np.array([[0, 0, 1, 0],
 [0, 0, 0, 1], [0, -9.8, -0.0614, 0], [0, 0, 0, 0]])  
B_lat = np.array([[0], [0], [0], [14.6122]])
C_lat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
D_lat = np.array([[0], [0]])

# for altitude
A_alt = np.array([[0, 1], [0, 0]])
B_alt = np.array([[0], [0.507614]])
C_alt = np.array([[1], [0]])
D_alt = np.array([[0]])

# for latitude
C_out_lat = np.array([1.0, 0.0, 0.0, 0.0])
A_lat_i = np.array([[0, 0, 1, 0, 0],
 [0, 0, 0, 1, 0], [0, -9.8, -0.0614, 0, 0], [0, 0, 0, 0, 0], [-1.0, 0.0, 0.0, 0.0, 0.0]]) 
B_lat_i = np.array([[0], [0], [0], [14.6122], [0.0]])

# for altitude
C_out_alt = np.array([1.0, 0.0, 0.0])
A_alt_i = [[0, 1, 0], [0, 0, 0], [-1.0, 0.0, 0.0]]
B_alt_i = np.array([[0.0], [0.507614], [0.0]])