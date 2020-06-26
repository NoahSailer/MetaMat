import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


filenames = np.array(['150/0.5mm.csv','150/1.5mm.csv','150/2.5mm.csv','150/chopped.csv','150/elevated.csv'])

def gaussian(x):
    std = 10*np.pi/180
    var = std**2.
    return np.exp(-(x**2.)/(2*var))/np.sqrt(2*np.pi*var)

def plot_cross_section(f):
    df = pd.read_csv(f)
    phi = df['Phi[deg]']
    theta = df['Theta[deg]']  
    g = df['abs(GainTotal)']
    theta0,g0=[],[]
    for i,p in enumerate(phi):
        if p==0:
            theta0.append(theta[i])
            g0.append(g[i])
    g0 = 10.*np.log10(np.array(g0))
    plt.plot(theta0,g0,label=f[4:-4])

def plot_profile():
    for f in filenames: plot_cross_section(f)
    plt.ylabel('Relative gain [dB]')
    plt.xlabel(r'Angle from boresight $\theta$ [deg]')
    plt.xlim(-5,40)
    plt.legend(loc=0)#,title=r'$\alpha_1$_$\alpha_2$_$\alpha_3$ [deg]')
    plt.title(r'$G(\theta,\phi=0)$')
    plt.show()

def integrated_gain(f,theta_max):
    theta_max = theta_max
    df = pd.read_csv(f)
    phi = df['Phi[deg]']
    theta = df['Theta[deg]']
    g = df['abs(GainTotal)']    
    phi0,theta0,g0 = [],[],[]
    for i,t in enumerate(theta):
        if t <= theta_max:
            phi0.append(phi[i])
            theta0.append(theta[i])
            g0.append(g[i])
    phi0 = np.array(phi0)*np.pi/180.
    theta0 = np.array(theta0)*np.pi/180.
    g0 = np.array(g0) #/g0[0] FOR NORMALIZED GAIN
    #BE CAREFUL, SPECIFY dtheta AND dphi
    #FOR EACH SIM
    dtheta,dphi=0.5*np.pi/180.,0.5*np.pi/180.
    s = 0
    for i in range(len(phi0)):s+=g0[i]*np.sin(theta0[i])*dtheta*dphi
    return s/(4.*np.pi)

def plot_integrated_gain():
    domain = np.linspace(0,20,40)
    for f in filenames:
        ig = [integrated_gain(f,x) for x in domain]
        plt.plot(domain,ig,label=f[4:-4])
    plt.ylabel('Integrated gain')
    plt.xlabel(r'Angle from boresight $\theta$ [deg]')
    plt.legend(loc=0)#,title=r'$\alpha_1$_$\alpha_2$_$\alpha_3$ [deg]')
    plt.title(r'$I(\theta) = \frac{1}{4\pi}\int_0^\theta \int_0^{2\pi} G(\psi,\phi)\sin(\psi)d\phi d\psi$')
    plt.show()

domain = np.linspace(0,20,40)
ig = [integrated_gain('150/chopped.csv',x) for x in domain]
plt.plot(domain,ig,label='chopped')
ig = [integrated_gain('150/elevated.csv',x) for x in domain]
plt.plot(domain,ig,label='elevated')
ig = [integrated_gain('150/0.5mm.csv',x) for x in domain]
plt.plot(domain,ig,label='oversized hex')
ig = [integrated_gain('150/outie.csv',x) for x in domain]
plt.plot(domain,ig,label='outie')
ig = [integrated_gain('150/mushroom_150_gain.csv',x) for x in domain]
plt.plot(domain,ig,label='mushroom')
plt.ylabel('Integrated gain (150 GHz)')
plt.xlabel(r'Angle from boresight $\theta$ [deg]')
plt.legend(loc=0)#,title=r'$\alpha_1$_$\alpha_2$_$\alpha_3$ [deg]')
plt.title(r'$I(\theta) = \frac{1}{4\pi}\int_0^\theta \int_0^{2\pi} G(\psi,\phi)\sin(\psi)d\phi d\psi$')
plt.savefig('integrated_gain_150.pdf',bbox_inches='tight')
plt.clf()



