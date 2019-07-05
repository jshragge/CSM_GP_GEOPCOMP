import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# define the function to perform trapezoidal integration
def trapez_int(fun,a,b,N,isfigure=False):
    ''' Function to perform trapezoidal integration.
    usage: val = trapez_int(fun,a,b,N,isfigure=False)
    input: 
        fun: function to be integrated
        a,b: integral interval
        N: number of subintervals 
        isfigure: plot the function and the nodes for demonstration
    written by Ge Jin, gjin@mines.edu, 06/2019
    '''
    h = (b-a)/N # calculate subinterval length
    xi = a+(np.arange(N+1))*h # calculate the location of nodes
    yi = fun(xi) # evaluate node values
    val = (np.sum(yi)-0.5*yi[0]-0.5*yi[-1])*h # calculate the result
    if isfigure:
        x = np.linspace(a,b,100)
        y = fun(x)
        plt.plot(x,y,label='Integrand')
        plt.plot(xi,yi,'ro',label='Nodes')
        plt.plot(xi,yi,'r-',label='Approx')
    return val


# define the function to perform Simpson's integration
def simpson_int(fun,a,b,N,isfigure=False):
    ''' Function to perform integration using Simpson's Rule.
    usage: val = simpson_int(fun,a,b,N,isfigure=False)
    input: 
        fun: function to be integrated
        a,b: integral interval
        N: number of subintervals 
        isfigure: plot the function and the nodes for demonstration
    written by Ge Jin, gjin@mines.edu, 06/2019
    '''
    N = N//2*2 # make sure N is a even number
    h = (b-a)/N # calculate subinterval length
    xi = a+(np.arange(N+1))*h # calculate the location of nodes
    yi = fun(xi) # evaluate node values
    s0 = yi[0]+yi[-1] # first and last nodes
    s1 = np.sum(yi[1:-1:2]) # odd number nodes
    s2 = np.sum(yi[2:-2:2]) # even number nodes
    val = h/3*(s0+4*s1+2*s2) # calculate the result
    if isfigure:
        x = np.linspace(a,b,100)
        y = fun(x)
        plt.plot(x,y,label='Integrand')
        plt.plot(xi,yi,'ro',label='Nodes')
        # plot Lagrange polynomial
        for i in range(N//2):
            x1 = xi[2*i+1]
            f = lambda s: 0.5*s*(s-1)*yi[2*i] - (s-1)*(s+1)*yi[2*i+1] + 0.5*(s+1)*s*yi[2*i+2]
            ind = (x>=xi[2*i])&(x<=xi[2*i+2])
            p = f((x[ind]-x1)/h)
            ax, = plt.plot(x[ind],p,'r')
        ax.set_label('Approx')
    return val


# define the function to perform Gauss quadrature
def gauss_int(fun,a,b,N,isfigure=False):
    ''' Function to perform Gauss quadrature
    usage: val = gauss_int(fun,a,b,N,isfigure=False)
    input: 
        fun: function to be integrated
        a,b: integral interval
        N: number of subintervals 
        isfigure: plot the function and the nodes for demonstration
    written by Ge Jin, gjin@mines.edu, 06/2019
    '''
    t, A = np.polynomial.legendre.leggauss(N)  
    xi = (b-a)/2*t+(a+b)/2   # calculate the location of nodes
    yi = fun(xi) # evaluate node values
    val = (b-a)/2*np.sum(A*yi)
    if isfigure:
        x = np.linspace(a,b,100)
        y = fun(x)
        ax = []
        ax += plt.plot(x,y,label='Integrand')
        ax += plt.plot(xi,yi,'ro',label='Gauss Nodes')
        plt.sca(plt.gca().twinx())
        ax += plt.bar(xi,A,width=0.1,alpha=0.5,color='r')
        labs = [l.get_label() for l in ax]
        labs[2] = 'Gauss Weights'
        plt.legend(ax[:3],labs[:3])
        plt.ylim([0,1])
    return val