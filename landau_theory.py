import numpy as np
import matplotlib.pyplot as plt

#defining constants
t0 = 373
a0 = 3.61

#define parameter functions
def a11(Tc):
    return -1.83*1e4 +40*Tc

def a111(Tc):
    return 1.39*1e5-3.2*100*Tc

def a12(Tc):
    return -2.24*1e4 + 6.7*10*Tc

def a112(Tc):
    return -2.2*1e4

def a123(Tc):
    return 5.51*1e5

#delta G function
def delta_G(px,py,pz,Tc):
    temp = 0.5*a0*(Tc - t0)*(px**2 + py**2 +pz**2)
    temp += 0.25*a11(Tc)*(px**4 + py**4 + pz**4)
    temp += 0.5*a12(Tc)*(px**2*py**2 + pz**2*py**2 + px**2*pz**2)
    temp += (a12(Tc)*(px**6 + py**6 + pz**6))/6
    temp += 0.5*a112(Tc)*(px**4*(pz**2 + py**2) + py**4*(px**2 + pz**2) + pz**4*(px**2 + py**2))
    temp += 0.5*a123(Tc)*(px**2*py**2*pz**2)
    return temp

#neglect the trivial solution
#del (delta G)/del px
def f1(px,py,pz,Tc):
    temp = a0*(Tc - t0)
    temp += a11(Tc)*px**2 + a12(Tc)*(py**2 + pz**2) + a111(Tc)*px**4 + a112(Tc)*2*(px*(py**2 + pz**2)) + a112(Tc)*(py**4 + pz**4) + a123(Tc)*(py**2*pz**2)
    return list(temp)[0]

#del(delat G)/del py
def f2(px,py,pz,Tc):
    temp = a0*(Tc - t0)
    temp += a11(Tc)*py**2 + a12(Tc)*(px**2 + pz**2) + a111(Tc)*py**4 + a112(Tc)*2*(py*(px**2 + pz**2)) + a112(Tc)*(px**4 + pz**4) + a123(Tc)*(px**2*pz**2)
    return list(temp)[0]

#del(delat G)/del pz
def f3(px,py,pz,Tc):
    temp = a0*(Tc - t0)
    temp += a11(Tc)*pz**2 + a12(Tc)*(px**2 + py**2) + a111(Tc)*pz**4 + a112(Tc)*2*(pz*(px**2 + py**2)) + a112(Tc)*(px**4 + py**4) + a123(Tc)*(px**2*py**2)
    return list(temp)[0]

#derivatives of f1
def f1_px(px,py,pz,Tc):
    return list(2*a11(Tc)*px + 4*a111(Tc)*px**3 + a112(Tc)*2*(pz**2 + py**2))[0]

def f1_py(px,py,pz,Tc):
    return list(2*a12(Tc)*py + 4*a112(Tc)*px*py + 4*a112(Tc)*py**3 + 2*a123(Tc)*py*pz**2)[0]

def f1_pz(px,py,pz,Tc):
    return list(2*a12(Tc)*pz + 4*a112(Tc)*px*pz + 4*a112(Tc)*pz**3 + 2*a123(Tc)*pz*py**2)[0]

#derivatives of f2
def f2_px(px,py,pz,Tc):
    return list(2*a12(Tc)*px + 4*a112(Tc)*px*py + 4*a112(Tc)*px**3 + 2*a123(Tc)*px*pz**2)[0]

def f2_py(px,py,pz,Tc):
    return list(2*a11(Tc)*py + 4*a111(Tc)*py**3 + a112(Tc)*2*(px**2 + pz**2))[0]

def f2_pz(px,py,pz,Tc):
    return list(2*a12(Tc)*pz + 4*a112(Tc)*pz*py + 4*a112(Tc)*pz**3 + 2*a123(Tc)*pz*px**2)[0]

#derivatives of f3
def f3_pz(px,py,pz,Tc):
    return list(2*a11(Tc)*pz + 4*a111(Tc)*pz**3 + a112(Tc)*2*(px**2 + py**2))[0]

def f3_px(px,py,pz,Tc):
    return list(2*a12(Tc)*px + 4*a112(Tc)*px*pz + 4*a112(Tc)*px**3 + 2*a123(Tc)*px*py**2)[0]

def f3_py(px,py,pz,Tc):
    return list(2*a12(Tc)*py + 4*a112(Tc)*py*pz + 4*a112(Tc)*py**3 + 2*a123(Tc)*py*px**2)[0]

#f = [f1_px,f1_py,f1_pz,f2_px,f2_py,f2_pz,f3_px,f3_py,f3_pz]

#Jacobian matrix
def J(px,py,pz,Tc):
    temp = [[f1_px(px,py,pz,Tc),f1_py(px,py,pz,Tc),f1_pz(px,py,pz,Tc)],[f2_px(px,py,pz,Tc),f2_py(px,py,pz,Tc),f2_pz(px,py,pz,Tc)],[f3_px(px,py,pz,Tc),f3_py(px,py,pz,Tc),f3_pz(px,py,pz,Tc)]]
    #for k in range(len(f)):
        #print(f[k](px,py,pz))
    return np.array(temp)

#functional
def F(px,py,pz,Tc):
    return np.array([[f1(px,py,pz,Tc)],[f2(px,py,pz,Tc)],[f3(px,py,pz,Tc)]])

if __name__ == '__main__':
    dg = []
    Tc = np.arange(370,391,0.5)
    for i in range(len(Tc)):
        x = np.array([[10],[10],[10]])
        itr = 300
        for j in range(itr):
            #j1 = J(x[0],x[1],x[2])
            #print(np.shape(j1))
            x = x - np.linalg.inv(J(x[0],x[1],x[2],Tc[i]))@F(x[0],x[1],x[2],Tc[i])
        dg.append(delta_G(x[0],x[1],x[2],Tc[i]))
        #print(f'Delta G = {delta_G(x[0],x[1],x[2],Tc[i])} for Tc = {Tc[i]}')
    dg = np.array(dg)
    plt.xlabel('$T_{c}$ (in Kelvin)')
    plt.ylabel("$\Delta G$")
    plt.scatter(Tc,dg)
    plt.show()

        
    
    
    



