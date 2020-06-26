import numpy as np
import matplotlib.pyplot as plt
import r
import sys

delta = 1.e-7 # step size for derivatives
alpha,beta = 2.e-2,1.e-8 
M = 5 # number of ARC layer + 2
endindex = 3.45 # silicon
minindex = 1.3 # minimum refractive index
trainsteps = 10000. # number of iterations
updatespeed,updatespeed1 = 2,int(trainsteps/20.) # how often I output the indices/widths
initial_thickness = 1.e-3 # initial total thickness of the ARC
fixedT = False # set to true if you want the total thickness of the ARC to the fixed
constraint = True # set to true if you want the lambda/4 constraint
angle_of_incidence = 0. 
bands = np.array([100.,145.,195.])
fbw = np.array([.23,.23,.23]) # fractional band width

for s in sys.argv:
    if s.startswith('initT='): 
        initial_thickness = float(s.replace('initT=',''))
        fixedT = True

freqs = np.arange(10.,260.,.5)
ks = 2.*np.pi*freqs*(1.e9)/299792458.

def plot_reflection(N,widths):
    A = []
    for k in ks:
        A.append(10.*np.log10(r.r(angle_of_incidence*np.pi/180.,k,N,widths)**2.))
    return A

def gauss(x,mu,sigma):
    return np.exp((-((x-mu)/sigma)**2.))

def ts(widths):
    w = np.copy(widths[1:-1])
    depths = np.zeros(len(w)+1)
    depths[0] = 0.
    depths[1] = w[0]
    for i in range(2,len(depths)): depths[i] = depths[i-1]+w[i-1]
    return depths

N = np.ones(M)
widths = np.ones(M)*initial_thickness/(M-2.)
N[-1] = endindex
for i in range(M):
    N[i] = minindex + (endindex-minindex)*((i/(len(N)-1.))**1.)
N[0] = 1.

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
ax.set_ylim(-70,0)
for i in range(6): ax.axhline(y=-10.*(i+1),color='k',linewidth=0.4,linestyle='--')

for i,f in enumerate(bands):
    ax.axvline(x=f,color='k',linewidth=0.4)
    ax.axvspan(f*(1.-fbw[i]/2.),f*(1.+fbw[i]/2.),alpha=0.1,color='k')
ax.set_xlabel('[GHz]')
ax.set_ylabel('Reflectance [dB]')
reflectivity, = ax.plot(freqs, plot_reflection(N,widths),color='k')
indices, = ax1.step(ts(widths)*1000.,[N[1]]+list(N[1:-1]),color='k')
ax1.set_ylabel('n')
ax1.set_xlabel('[mm]')
ax1.grid()
fig.show()

def f(N,widths):
    '''
    Returns the loss function for an array of indices N and 
    widths. 
    '''
    loss = 0.
    for i,f in enumerate(bands):
        band = np.arange(f*(1.-fbw[i]/2.),f*(1.+fbw[i]/2.),2.)
        kband = 2.*np.pi*band*(1.e9)/299792458.
        x = np.mean(np.array([r.r(angle_of_incidence*np.pi/180.,k,N,widths)**2. for k in kband]))
        loss += x
    return loss

for i in range(int(trainsteps)):
    '''
    Here's where you repeatedly take derivatives.
    '''
    gradient,gradient1 = np.zeros(M),np.zeros(M)
    for j in range(M-2):
        dvar = np.zeros(M)
        dvar[j+1] = delta
        gradient[j+1] = (f(N+dvar,widths)-f(N,widths))/delta
        gradient1[j+1] = (f(N,widths+dvar)-f(N,widths))/delta
    N -= alpha*gradient
    widths -= beta*gradient1
    N = np.sort(N)
    if constraint:
        lam = 299792458./(np.mean(bands)*(1.e9))
        widths = lam/(4.*N)
    for k in range(1,M-1):
        if N[k] > endindex:
            N[k] = endindex
        elif N[k] < minindex:
            N[k] = minindex
        if widths[k] < 0.:
            widths[k] = 1.e-3
    if fixedT: 
        widths = widths*initial_thickness/sum(widths[1:-1])
        widths[0],widths[-1] = 0.00000001,0.000000001
    if i%updatespeed == 0: 
        reflectivity.set_ydata(plot_reflection(N,widths))
        indices.set_ydata([N[1]]+list(N[1:-1]))
        indices.set_xdata(ts(widths)*1000.)
        ax.set_title(str(i*100/trainsteps)+' %')
        ax1.set_xlim(-0.01,ts(widths)[-1]*1000.+0.01)
        ax1.set_ylim(0.8,endindex+0.2)
        fig.canvas.draw()
        fig.canvas.flush_events()
    if i%updatespeed1 == 0:
        print str(i*100/trainsteps),'% -----------------------------------------------------------------'
        print 'N = np.'+repr(N)
        print 'widths = np.'+repr(widths)
        print 'thickness =',sum(widths[1:-1])*1000.
