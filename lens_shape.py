import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###################################################################################

class lens:

	def __init__(s, resolution=5000, phiMax=90., alphaMax=31., F=2., S=5.,
	       n1=1.6, n2=2.3, phi0=64.*np.pi/180., alpha0 =23.*np.pi/180.):
		
		s.resolution = resolution
		s.phiMax = phiMax
		s.alphaMax = alphaMax
		s.F = F
		s.S = S
		s.n1 = n1
		s.n2 = n2
		s.phi0 = phi0
		s.alpha0 = alpha0
		s.phi = np.linspace(0.,phiMax * np.pi/180., resolution)
		s.alpha = np.linspace(0.,alphaMax * np.pi/180., resolution)
		s.r = None
		s.R = None
		s.K = 0.
		s.alphaFromPhi = None


	#this is temporary, fix with actual function
	def T(s,phi): return 1.


	def U(s,phi): return np.exp(-2. * (phi / s.phi0)**2.)


	def G(s,alpha): return np.exp(-2. * (alpha / s.alpha0)**2.)


	def computeK(s,phi,alpha):
		dphi = phi[1] - phi[0]
		dalpha = alpha[1] - alpha[0]
	
		numerator = np.sum(s.T(phi) * s.U(phi) * np.sin(phi) * dphi)
		denomenator = np.sum(s.G(alpha) * np.sin(alpha) * dalpha)

		s.K = numerator/denomenator


	def dalpha_dphi(s,phi, alpha, K):
		if alpha == 0.: return np.sqrt( s.T(0.) * s.U(0.) / (K * s.G(0.)) )
		return s.T(phi) * s.U(phi) * np.sin(phi) / (K * s.G(alpha) * np.sin(alpha))


	def computeAlphaFromPhi(s,phi, K, resolution):
		dphi = phi[1] - phi[0]
		alphaFromPhi = np.zeros(resolution)
	
		for i in range(1,resolution):
			alphaFromPhi[i] = alphaFromPhi[i-1] + s.dalpha_dphi(phi[i-1], s.alpha[i-1], K) * dphi

		s.alphaFromPhi = alphaFromPhi
		return alphaFromPhi

	def dr_dphi(s,phi, gamma, r, n1, n2):
		return r * np.sin(phi - gamma) / (np.cos(phi - gamma) - n2/n1)


	def dR_dphi(s,phi, alpha, gamma, R, n1):
		if alpha == 0.:
			numerator = R * n1 * np.sin(gamma - alpha) * np.sqrt( s.T(phi) * s.U(phi) )
			denomenator = (1. - n1 * np.cos(gamma - alpha)) * np.sqrt( s.K * s.G(alpha) )
			return numerator / denomenator

		numerator = R * n1 * np.sin(gamma - alpha) * s.T(phi) * s.U(phi) * np.sin(phi)
		denomenator = (1. - n1 * np.cos(gamma - alpha)) * (s.K * s.G(alpha) * np.sin(alpha))
		return numerator/denomenator


	def gamma(s,phi, alpha, n1, n2, R, r, F, S):
		z0 = F * (n2 - 1.) + S * (n1 -1.)
		numerator = n1 * (R * np.cos(alpha) - r * np.cos(phi) - z0)	
		denomenator = R - n2 * r
		if phi == 0.: return 0.
		if numerator/denomenator > 1.: 
			return 0.
		return np.arccos(numerator/denomenator)


	def computeRandR(s,phi, alphaFromPhi, K, F, S, n1, n2, resolution):
		dphi = phi[1] - phi[0]
		r,R = np.zeros(resolution),np.zeros(resolution)
		r[0],R[0] = F, 	n2 * F + n1 * S

		for i in range(1, resolution):
			g = s.gamma(phi[i-1], alphaFromPhi[i-1], n1, n2, R[i-1], r[i-1], F, S)
			r[i] = r[i-1] + s.dr_dphi(phi[i-1], g, r[i-1], n1, n2) * dphi
			R[i] = R[i-1] + s.dR_dphi(phi[i-1], alphaFromPhi[i-1], g, R[i-1], n1) * dphi

		s.r = r
		s.R = R


	def solveLens(s):
		s.computeK(s.phi, s.alpha)
		alphaFromPhi = s.computeAlphaFromPhi(s.phi, s.K, s.resolution)
		s.computeRandR(s.phi, alphaFromPhi, s.K, s.F, s.S, s.n1, s.n2, s.resolution)

	def plot(s,c='k'):
		inner_x, inner_y = s.r * np.sin(s.phi), s.r * np.cos(s.phi)
		outer_x, outer_y = s.R * np.sin(s.alphaFromPhi), s.R * np.cos(s.alphaFromPhi) - s.F*(s.n2-1.)-s.S*(s.n1-1.)	

                a,b = 93, 92
                inner_x, inner_y = inner_x[:a], inner_y[:a]
                outer_x, outer_y = outer_x[:b], outer_y[:b]
              	
		plt.plot(inner_x, inner_y, c=c, lw=2, label = str(s.F) + " " + str(s.S))
		plt.plot(-inner_x, inner_y, c=c, lw=2)
		plt.plot(outer_x, outer_y, c=c, lw=2)
		plt.plot(-outer_x, outer_y, c=c, lw=2)	

###################################################################################

norm = plt.Normalize()
colors = plt.cm.Greens(norm([1,2,3,4,5,6]))

#l0 = lens(n2 = 2.8)
#l0.solveLens()
#l0.plot(c=colors[1])

#l1 = lens(n2 = 3.0)
#l1.solveLens()
#l1.plot(c=colors[2])

#l2 = lens(n2 = 3.2)
#l2.solveLens()
#l2.plot(c=colors[3])

#l3 = lens(n2 = 3.4)
#l3.solveLens()
#l3.plot(c=colors[4])


l4 = lens(n1 = 1.5, n2 = 3.45, F = 2.2, S=7.25, resolution=100)
l4.solveLens()
l4.plot(c=colors[4])


plt.ylim(0,10)
plt.xlabel('mm')
plt.ylabel('mm')
plt.legend(loc=2,title="Inner index")
plt.show()


def make_csv(l):
   inner_x, inner_y = l.r * np.sin(l.phi), l.r * np.cos(l.phi)
   outer_x, outer_y = l.R * np.sin(l.alphaFromPhi), l.R * np.cos(l.alphaFromPhi) - l.F*(l.n2-1.)-l.S*(l.n1-1.)

   a,b = 92, 92
   inner_x, inner_y = inner_x[:a], inner_y[:a]
   outer_x, outer_y = outer_x[:b], outer_y[:b]
   z = np.zeros(len(inner_x))

   df_inner_pos = pd.DataFrame({'x': inner_x,
                   'y': inner_y,
                   'z': z})

   df_inner_neg = pd.DataFrame({'x': -inner_x,
                   'y': inner_y,
                   'z': z})

   df_outer_pos = pd.DataFrame({'x': outer_x,
                   'y': outer_y,
                   'z': z})

   df_outer_neg = pd.DataFrame({'x': -outer_x,
                   'y': outer_y,
                   'z': z})

   np.savetxt('inner_pos.txt', df_inner_pos.values, fmt='%1.3f')
   np.savetxt('inner_neg.txt', df_inner_neg.values, fmt='%1.3f')
   np.savetxt('outer_pos.txt', df_outer_pos.values, fmt='%1.3f')
   np.savetxt('outer_neg.txt', df_outer_neg.values, fmt='%1.3f')

make_csv(l4)
