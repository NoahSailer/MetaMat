import numpy as np
import cmath as c
import matplotlib.pyplot as plt

def trajectory(theta,k,N):

    '''Given an initial incident angle and 
    wavenumber, this returns a list of angles and 
    wavevectors as the wave passes through the
    layers'''
    
    M = len(N)
    THETA,KX = np.zeros(M),np.zeros(M)
    for i,n in enumerate(N):
        KX[i] = np.sqrt(n**2.-np.sin(theta)**2.)*k
    if theta == 0:
        THETA = np.zeros(M)
    else:
        THETA = np.ones(M)*(np.pi/2.)-np.arctan(KX/(k*np.sin(theta)))
    return THETA,KX


def p(kx,d):
    
    '''returns a matrix that propogates the wave
    through a layer with thickness d and normal
    component to the wavevector kx'''

    return np.array([[c.exp(complex(0,kx*d)),0],
                     [0,c.exp(complex(0,-kx*d))]])


def t(n,m,theta):
    
    '''returns the matrix T_n,m which propogates
    the wave across a layer from index n and 
    incident angle theta to a layer with index m
    and exiting angle phi.'''
    
    phi = np.arcsin(n*np.sin(theta)/m)
    A = n*np.cos(theta)/(2.*m*np.cos(phi))
    return np.array([[0.5+A,0.5-A],
                     [0.5-A,0.5+A]])


def D(theta,k,N,widths):

    '''This returns the matrix D, which gives
    you the relationship between the fields
    (E_{M-1},E'_{M-1}) and (E_1, E'_1)'''

    M = len(N)
    THETA,KX = trajectory(theta,k,N)
    D = np.array([[1,0],[0,1]])
    for i in range(M-3):
        D = np.dot(t(N[i],N[i+1],THETA[i]),D)
        D = np.dot(p(KX[i+1],widths[i+1]),D)
    return np.dot(t(N[-3],N[-2],THETA[-3]),D)


def r(theta,k,N,widths):
    
    '''returns the transmission coefficient for an
    incoming wave with incident angle theta and 
    wavenumber k on an M-2 layer coating.'''
    
    THETA,KX = trajectory(theta,k,N)
    trm = D(theta,k,N,widths)
    D11,D12,D21,D22 = trm[0,0],trm[0,1],trm[1,0],trm[1,1]
    A,B = N[-1]*np.cos(THETA[-1]),N[-2]*np.cos(THETA[-2])
    C = c.exp(complex(0,KX[-2]*widths[-2]))
    M = np.array([[1.,-C  ,-1./C,],
                  [A ,-B*C, B/C  ],
                  [0 ,D22 ,-D12  ]])
    V = np.array([0,0,D11*D22-D12*D21])
    r = np.dot(np.linalg.inv(trm),np.dot(np.linalg.inv(M),V)[1:])[1]
    #t = np.dot(np.linalg.inv(M),V)[0]
    return abs(r)
