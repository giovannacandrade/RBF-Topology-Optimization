import numpy as np
from dolfin import *
from matplotlib import cm,pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
scipy.sparse.linalg.use_solver()

class PLSF:
    def __init__(self, phi_mat, dsp, lx, ly, X, Y, nNodes):
        self.mat = phi_mat
        self.inner = 1
        self.dsp = dsp; self.c = 10**(-2)
        self.lx, self.ly = lx, ly;  self.X, self.Y = X, Y; self.nNodes = nNodes
        self.interpol()
        self.NPx, self.NPy = [180,60]
        self.XP,self.YP = np.meshgrid(np.linspace(0.0,lx,self.NPx+1),np.linspace(0.0,ly,self.NPy+1)); 
        self.Gp = self.transf_matrix(self.X.flatten('F'),self.Y.flatten('F'),self.nNodes,self.XP.flatten('F'),self.YP.flatten('F'),(self.NPx + 1)*(self.NPy+1),0)
  
    def plot(self,It):
        phi_mat = np.reshape(csr_matrix.dot(self.Gp,self.alpha),(self.NPy+1,self.NPx+1),order='F')
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5) 
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.XP, self.YP, phi_mat, rstride=1, cstride=1, cmap = 'coolwarm', shade = True, linewidth=0, alpha = 0.8) # ax.set_box_aspect((4,2,0.8))
        ax.elev = 55
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.locator_params(axis='both',  nbins=5)#         plt.show()
        plt.savefig('Surf_it_'+str(It)+'.pdf',bbox_inches='tight')
        plt.figure(figsize=(7, 7))
        cs = plt.subplot()  
        cs.contourf(phi_mat, levels=[0,np.max(phi_mat)],extent = [.0,self.lx,.0,self.ly],cmap=cm.get_cmap('Accent'))
        cs.set_aspect('equal','box')
        cs.set_aspect('equal','box')
        plt.axis('off')
        plt.savefig('Contourf_it_'+str(It)+'.pdf',bbox_inches='tight')    
        plt.show()
        return    
    
    def transf_matrix(self,X,Y,nNodes,X_,Y_,nNodes_,k):
        Ax = X_.reshape(nNodes_,1) - X.reshape(1,nNodes)
        Ay = Y_.reshape(nNodes_,1) - Y.reshape(1,nNodes)
        r_CS = np.sqrt(Ax**2+Ay**2+self.c**2)/self.dsp      
        G_ = csr_matrix(((np.maximum(0,1-r_CS))**4)*(4*r_CS + 1),dtype = 'float64')
        if k == 1:
            return  G_, prpX1, prpX2
        else:
            return G_
    
    
    def interpol(self):   
        Ax = self.X.reshape(self.nNodes,1,order='F') - self.X.reshape(1,self.nNodes,order='F')
        Ay = self.Y.reshape(self.nNodes,1,order='F') - self.Y.reshape(1,self.nNodes,order='F')
        r_CS = np.sqrt(Ax**2+Ay**2+self.c**2)/self.dsp      
        self.G = csr_matrix(((np.maximum(0,1-r_CS))**4)*(4*r_CS + 1),dtype = 'float64')
        self.alpha = spsolve(self.G,self.mat.flatten('F'))
        self.prpX1_ = csr_matrix((np.maximum(0,1-r_CS)**3)*(-20*r_CS)*(Ax/r_CS/self.dsp**2),dtype = 'float64')
        self.prpX2_ = csr_matrix((np.maximum(0,1-r_CS)**3)*(-20*r_CS)*(Ay/r_CS/self.dsp**2),dtype = 'float64')
        return
 
    def evolve(self, th, lx, ly, Nx, Ny, dt, SD):       
        if SD == 'HB' or SD == 'natural':
            self.delta = 0.5
            for i in range(self.inner):
                indexDelta = (abs(self.mat)<=self.delta)
                DeltaPhi = np.zeros((Ny+1,Nx+1))
                DeltaPhi[indexDelta] = 1/(2*self.delta) + 1/(2*self.delta)*np.cos(np.pi*self.mat[indexDelta]/self.delta)
                B = th*DeltaPhi.flatten('F')*self.delta
                dAlpha = spsolve(self.G,B)
                self.alpha = self.alpha + dt*dAlpha
                self.mat = np.reshape(csr_matrix.dot(self.G,self.alpha),(Ny+1,Nx+1),order='F')
            return
        elif SD == 'HV':
            for i in range(self.inner):
                nNodes = (Nx+1)*(Ny+1)
                th_x1 = th[0].flatten('F')
                th_x2 = th[1].flatten('F')
                aux = np.zeros((nNodes,nNodes))
                for i in range(nNodes):
                    aux[i] = (th_x1[i]*self.prpX1_[i,:] + th_x2[i]*self.prpX2_[i,:]).toarray()
                W = np.dot(aux,self.alpha)
                dAlpha = spsolve(self.G, W)
                self.alpha = self.alpha - dt*dAlpha
                self.mat = np.reshape(csr_matrix.dot(self.G,self.alpha),(Ny+1,Nx+1),order='F')
            return