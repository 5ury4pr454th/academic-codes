# this file for finding the best rise time without any input saturation
import specs as p

def return_ks(t_rise_alt, t_rise_lateral):
    
    t_lat_inner = t_rise_lateral/10

    w_lat_inner = 2.2/t_lat_inner
    w_lateral = 2.2/t_rise_lateral
    w_alt = 2.2/t_rise_alt

    b0_alt = 1/(p._m_c + p._m_l + p._m_r)
    b0_lat = -9.8
    b1_lat = p._m_u/(p._m_c + p._m_l + p._m_r)

    damping_constant = 0.707
    # Now, for finding best kp and kd:

    kd_alt = 2 * damping_constant * w_alt/b0_alt
    kp_alt = w_alt**2/b0_alt

    kdc_lat = 1/((w_lat_inner**2)*(p._J_c + (p._m_l + p._m_r)*p._d_br**2))

    kp_lat = w_lateral**2/(b0_lat*kdc_lat)
    kd_lat = ((2 * damping_constant * w_lateral) - b1_lat)/(b0_lat*kdc_lat)

    return kd_alt, kp_alt, kdc_lat, kd_lat, kp_lat
