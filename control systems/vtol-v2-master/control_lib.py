import control as cnt
import numpy as np
import state_specs as sp
import specs as p

# calculate controllability matrix
ctrb_matrix_alt = cnt.ctrb(sp.A_alt, sp.B_alt)
ctrb_matrix_lat = cnt.ctrb(sp.A_lat, sp.B_lat)

# checking for controlability
if np.linalg.matrix_rank(ctrb_matrix_alt) == sp.A_alt.shape[0]:
    print('Altitude system is controllable...')
else:
    print("The Altitude system is not controllable")

if np.linalg.matrix_rank(ctrb_matrix_lat) == sp.A_lat.shape[0]:
    print('Latitude system is controllable...')
else:
    print("The Latitude system is not controllable")

def gains_from_time(t_rise, alt = True):
    wn = 2.2 / t_rise
    if alt==True:
        desired_poles = np.roots([1, 2*wn*p._damp, wn**2])
        K = cnt.acker(sp.A_alt, sp.B_alt, poles = desired_poles)
        print(sp.C_alt.shape, sp.A_alt.shape, sp.B_alt.shape, K.shape)
        kr = -1.0/(sp.C_alt.T @ np.linalg.inv(sp.A_alt - sp.B_alt @ K) @ sp.B_alt)
    else:
        desired_poles = np.roots(np.convolve([1, 2*p._damp*wn*10, 100*(wn**2)], [1, 2*p._damp*wn, wn**2]))
        K = cnt.acker(sp.A_lat, sp.B_lat, poles = desired_poles)
        Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
        kr = -1.0/(Cr @ np.linalg.inv(sp.A_lat - sp.B_lat @ K) @ sp.B_lat)
    return K, np.squeeze(kr)

# K_alt_11 = cnt.place(sp.A_alt, sp.B_alt, np.array([-0.24 + 0.18j, -0.24 - 0.18j]))
# K_lat_11 = cnt.place(sp.A_lat, sp.B_lat, np.array([-0.24 + 0.18j, -0.24 - 0.18j]))

# kr_lat_11 = -1.0/(ctrb_matrix_lat @ np.linalg.inv(sp.A_lat - sp.B_lat @ K_lat_11) @ sp.B_lat)
# kr_alt_11 = -1.0/(ctrb_matrix_alt @ np.linalg.inv(sp.A_alt - sp.B_alt @ K_alt_11) @ sp.B_alt)

def integral_gains_from_time(t_rise, pi, alt = True):
    wn = 2.2 / t_rise

    if alt==True:
        desired_poles = np.roots(np.convolve([1, 2*wn*p._damp, wn**2], np.poly(pi)))
        K = cnt.acker(sp.A_alt_i, sp.B_alt_i, poles = desired_poles)
    else:
        desired_poles = np.roots(np.convolve(np.convolve([1, 2*p._damp*wn, (wn**2)], [1, 2*p._damp*wn*10, 100*(wn**2)]), np.poly(pi)))
        K = cnt.place(sp.A_lat_i, sp.B_lat_i, desired_poles)
    return K[0,0:-1], K[0, -1]