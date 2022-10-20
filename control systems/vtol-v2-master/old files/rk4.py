import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace


# def dfdx(x,t):
#     return x - np.sin(t)

# def inte(x,t):
#     return x*t + np.cos(t)-1

def dfdx(x,t):
    return x*t

def inte(x,t):
    return x*(t**2)/2 + 1 # + 1 to make the graph visible

t = 0
dt = 0.01
t1 = -1
dt1 = 0.01
f0 = 1
fx = [f0]
tm = [t]
tm1 = [t1]
f = f0
intfx = [inte(f0,t1)]
count = 0
fig = plt.figure()


plt.ion()

while t<10:

    x = linspace(0,1,100) # this is the line which you have to draw for the slope

    # prepares previous quadratic function
    # for every small increment in t, the following is plotted again and again
    while t1<t: 
        t1 += dt1 # t1 is time before t
        tm1.append(t1) # tm1 keeps track of all times
        intfx.append(inte(f0,t1)) # intfx keeps track of all y values
    plt.plot(tm1,intfx,c='purple')

    # first slope
    # f0 is the value of f at f = t
    k1 = dfdx(f0,t) # dfdx(f0,t) gives the derivative of f for value of f when t = t. Derivative of
    c1 = plt.plot(x+t,f+k1*x,c='blue') # then plot the line whose equation is given by f+k1* 
    s1 = plt.scatter(t,f,c='blue')

    # plt.pause(0.1)
    k2 = dfdx(f0+dt*k1/2,t+dt/2)
    c2 = plt.plot(x+t+dt/2,f+dt*k1/2+k2*x,c='green')
    s2 = plt.scatter(t+dt/2,f+dt*k1/2,c='green')

    # plt.pause(0.1)
    k3 = dfdx(f0+dt*k2/2,t+dt/2)
    c3 = plt.plot(x+t+dt,f+dt*k1/2+ dt*k2/2+k3*x,c='black')
    s3 = plt.scatter(t+dt,f+dt*k1/2+dt*k2/2,c='black')

    # plt.pause(0.1)
    k4 = dfdx(f0+dt*k3,t+dt)
    c4 = plt.plot(x+t+3*dt/2,f+dt*k1/2+ dt*k2/2+ dt*k3/2+k4*x,c='brown')
    s4 = plt.scatter(t+3*dt/2,f+dt*k1/2+dt*k2/2+dt*k3/2,c='brown')

    # plt.pause(0.1)
    f += dt*(k1+2*k2+2*k3+k4)/6
    t += dt
    
    fx.append(f)
    tm.append(t)
    plt.plot(tm,fx,color = 'red')
    
    plt.pause(0.01)
    c1.pop(0).remove()
    s1.remove()
    c2.pop(0).remove()
    s2.remove()
    c3.pop(0).remove()
    s3.remove()
    c4.pop(0).remove()
    s4.remove()
    fig.canvas.flush_events()
    plt.show()

