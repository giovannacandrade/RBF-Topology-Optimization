import numpy as np
from dolfin import *
from matplotlib import cm,pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for 3d plotting 

class Structure:
    def __init__(self, name, lx, ly, Nx, Ny, E, nu, eps_er, velocity_ext,a):
        self.name = name
        self.Nx, self.Ny = Nx, Ny
        self.lx, self.ly = lx, ly
        self.mu, self.lmbda = Constant(E/(2*(1 + nu))),Constant(E*nu/((1+nu)*(1-2*nu)))
        self.lmbda =  Constant(2*self.mu*self.lmbda/(self.lmbda+2*self.mu))
        self.eps_er = eps_er
        self.a = a
      
        if name == 'cantilever1':
            self.mesh  = RectangleMesh(Point(0.0,0.0),Point(lx,ly),Nx,Ny,'crossed')
            self.Vvec = VectorFunctionSpace(self.mesh, 'CG', 1) 
            class DirBd(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0],.0)
            dirBd = DirBd()
            boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
            boundaries.set_all(0); dirBd.mark(boundaries, 1)
            self.ds = Measure("ds")(subdomain_data=boundaries)  
            self.bcd  = [DirichletBC(self.Vvec, (0.0,0.0), boundaries, 1)] 
            self.Load = [Point(lx, ly*0.5)]
        
        elif name == 'cantilever2':
            self.mesh  = RectangleMesh(Point(0.0,0.0),Point(lx,ly),Nx,Ny,'crossed')
            self.Vvec = VectorFunctionSpace(self.mesh, 'CG', 1) 
            class DirBd(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0],.0)
            dirBd = DirBd()
            boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
            boundaries.set_all(0); dirBd.mark(boundaries, 1)
            self.ds = Measure("ds")(subdomain_data=boundaries)  
            self.bcd  = [DirichletBC(self.Vvec, (0.0,0.0), boundaries, 1)] 
            self.Load = [Point(lx, 0.0)]   

        elif name == 'half_wheel':
            self.mesh  = RectangleMesh(Point(0.0,0.0),Point(lx,ly),Nx,Ny,'crossed') 
            self.Vvec = VectorFunctionSpace(self.mesh, 'CG', 1)  
            tol = 1e-8
            class DirBd(SubDomain):
                def inside(self, x, on_boundary):
                    return abs(x[0])< tol and abs(x[1])< tol 
            class DirBd2(SubDomain):
                def inside(self, x, on_boundary):
                    return abs(x[0]-lx)< tol and abs(x[1])< tol
            dirBd,dirBd2 = [DirBd(),DirBd2()]
            boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
            boundaries.set_all(0)          
            dirBd.mark(boundaries, 1)
            dirBd2.mark(boundaries, 2)  
            self.ds = Measure("ds")(subdomain_data=boundaries)                     
            self.Load = [Point(lx/2, 0.0)]
            self.bcd  = [DirichletBC(self.Vvec, (0.0,0.0), dirBd, method='pointwise'),\
               DirichletBC(self.Vvec.sub(1), 0.0, dirBd2,method='pointwise')]      

        elif name == 'bridge':
            self.mesh  = RectangleMesh(Point(0.0,0.0),Point(lx,ly),Nx,Ny,'crossed') 
            self.Vvec = VectorFunctionSpace(self.mesh, 'CG', 1)  
            tol = 1e-8
            class DirBd(SubDomain):
                def inside(self, x, on_boundary):
                    return abs(x[0])< tol and abs(x[1])< tol 
            class DirBd2(SubDomain):
                def inside(self, x, on_boundary):
                    return abs(x[0]-lx)< tol and abs(x[1])< tol
            dirBd,dirBd2 = [DirBd(),DirBd2()]
            boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
            boundaries.set_all(0)          
            dirBd.mark(boundaries, 1)
            dirBd2.mark(boundaries, 2)
            self.ds = Measure("ds")(subdomain_data=boundaries)                     
            self.Load = [Point(lx/2, 0.0)]
            self.bcd  = [DirichletBC(self.Vvec, (0.0,0.0), dirBd, method='pointwise'),\
               DirichletBC(self.Vvec, (0.0,0.0), dirBd2, method='pointwise')]       
        elif name == 'MBB_beam':  
            tol = 1e-8
            self.mesh  = RectangleMesh(Point(0.0,0.0),Point(lx,ly),Nx,Ny,'crossed') 
            self.Vvec = VectorFunctionSpace(self.mesh, 'CG', 1)                
            class DirBd(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0],.0)   
            class DirBd2(SubDomain):
                def inside(self, x, on_boundary):
                    return abs(x[0]-lx) < tol and abs(x[1]) < tol    
            dirBd,dirBd2 = [DirBd(),DirBd2()]
            boundaries = MeshFunction("size_t",self.mesh, self.mesh.topology().dim() - 1)
            boundaries.set_all(0)          
            dirBd.mark(boundaries, 1)
            dirBd2.mark(boundaries, 2)  
            self.ds = Measure("ds")(subdomain_data=boundaries)
            self.bcd  = [DirichletBC(self.Vvec.sub(0), 0.0, boundaries, 1),\
                    DirichletBC(self.Vvec.sub(1), 0.0, dirBd2,method='pointwise')]                        
            self.Load = [Point(0.0, ly)] 

        #Function Space
        self.V = FunctionSpace(self.mesh, 'CG', 1)   
        # Unit Cell Volume
        self.VolUnit = project(Expression('1.0',degree=2),self.V) 
        # Displacement Vector
        self.U = [0]*len(self.Load)
        # Vertices Coordinates
        self.dofsV    = self.V.tabulate_dof_coordinates().reshape((-1, self.mesh.geometry().dim()))    
        self.dofsVvec = self.Vvec.tabulate_dof_coordinates().reshape((-1, self.mesh.geometry().dim()))  
        self.px, self.py    = [(self.dofsV[:,0]/lx)*2*Nx, (self.dofsV[:,1]/ly)*2*Ny]
        self.pxvec, self.pyvec = [(self.dofsVvec[:,0]/lx)*2*Nx, (self.dofsVvec[:,1]/ly)*2*Ny]  
        self.dofsV_max, self.dofsVvec_max =((Nx+1)*(Ny+1) + Nx*Ny)*np.array([1,2]) 
        self.dX = Measure('dx')
        self.n = FacetNormal(self.mesh)
       
        # Shape Derivative 
        if velocity_ext == 'HB':
            theta,xi = [TrialFunction(self.V), TestFunction(self.V)]     
            self.solverav = LUSolver(self._inner_product(velocity_ext, theta, xi,a))    
        if velocity_ext == 'HV':
            theta,xi = [TrialFunction(self.Vvec), TestFunction(self.Vvec)]    
            self.solverav = LUSolver(self._inner_product(velocity_ext, theta, xi,a))

    def _map_geometry(self, H, d, g):
        self.heav = Function(FunctionSpace(self.mesh, 'CG', 1))
        self.heav.vector().set_local(H)
        self.delta = Function(FunctionSpace(self.mesh, 'CG', 1))
        self.delta.vector().set_local((1 - self.eps_er)*d)
        self.grad = Function(FunctionSpace(self.mesh, 'CG', 1))
        self.grad.vector().set_local(g)
        self.rho = Function(FunctionSpace(self.mesh, 'CG', 1))
        self.rho.vector().set_local(self.eps_er + (1 - self.eps_er)*H)
        return
 
    def _get_compliance(self):
        compliance = 0
        for u in self.U:
            eU = sym(grad(u))
            S1 = 2.0*self.mu*inner(eU,eU) + self.lmbda*div(u)*div(u)
            compliance += assemble(S1*self.rho*self.dX)  
        return compliance
          
    def _get_volume(self):
        self.vol = assemble(self.heav*self.dX)/(self.lx*self.ly)
        return self.vol
    
    def SolveLinearElasticity(self):
        u,v = [TrialFunction(self.Vvec), TestFunction(self.Vvec)]
        S1 = 2.0*self.mu*inner(sym(grad(u)),sym(grad(v))) + self.lmbda*div(u)*div(v)
        A = assemble(S1*self.rho*self.dX) 
        b = assemble(inner(Expression(('0.0', '0.0'),degree=2) ,v)*self.ds(2)) 
        for k in range(0,len(self.Load)):   
            aux = Function(self.Vvec)
            delta = PointSource(self.Vvec.sub(1), self.Load[k], -1.0)
            delta.apply(b) 
            for bc in self.bcd: bc.apply(A,b)    
            solver = LUSolver(A)
            solver.solve(aux.vector(), b)    
            self.U[k] = aux
        return

    def _inner_product(self, velocity_ext, theta, xi,a):
        if velocity_ext == 'HB':
            self.a = a; self.b = 1.0; self.c = 0
            return assemble((self.a*inner(grad(theta),grad(xi)) + self.b*inner(theta,xi))*dX )
        if velocity_ext == 'HV':
            self.a = a; self.b = 1.0; self.c = 1e5
            return assemble((self.a*inner(grad(theta),grad(xi)) + self.b*inner(theta,xi))*self.dX +\
             self.c*(inner(dot(theta,self.n),dot(xi,self.n))* (self.ds(0)+ self.ds(1)+ self.ds(2)))) 
                         
    def ShapeDerivative(self, Lag, rho, volfrac, velocity_ext,zeta):
        rv = 0.0 
        if velocity_ext == 'HB':
            xi = TestFunction(self.V); Vn = Function(self.V) 
            for u in self.U:       
                G = - 2.0*self.mu*inner(sym(grad(u)),sym(grad(u))) - self.lmbda*div(u)*div(u) 
                S = G + Constant(rho*np.maximum(0,Lag/rho + zeta*(self.vol - volfrac)))
                rv =  - assemble(inner((1-self.eps_er)*S,xi)*self.delta*self.grad*self.dX)
            self.solverav.solve(Vn.vector(), rv)
            self.Vn =  Vn
            return
        elif velocity_ext == 'HV':
            xi = TestFunction(self.Vvec); th = Function(self.Vvec)                   
            for u in self.U:       
                eu,Du,Dxi = [sym(grad(u)),grad(u),grad(xi)]
                S1 = 2*self.mu*(2*inner((Du.T)*eu,Dxi) - inner(eu,eu)*div(xi))\
                 + self.lmbda*(2*inner( Du.T, Dxi )*div(u) - div(u)*div(u)*div(xi) )
                rv += - assemble(S1*self.rho*self.dX + \
                 Constant((rho*np.maximum(0,Lag/rho + zeta*(self.vol - volfrac))))*div(xi)*self.rho*self.dX)
            self.solverav.solve(th.vector(), rv)
            self.th = th
            return 
        elif velocity_ext == 'natural':
            eU = sym(grad(self.U[0]))
            S1 = 2.0*self.mu*inner(eU,eU) + self.lmbda*div(self.U[0])*div(self.U[0]) 
            V0 = FunctionSpace(self.mesh,"DG",0)
            eleComp = project(S1*self.rho,V0) #for each triangular element
            self.Vn = eleComp
            return