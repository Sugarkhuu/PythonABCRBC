# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 22:46:27 2018

@author: Sugarkhuu
"""

import numpy as np
import math



theta=.36
beta=.99
delta=.025
A=1.72
kbar=12.6695
hbar=.3335
ybar=kbar**theta*hbar**(1-theta)
cbar=ybar-delta*kbar
aa=(theta*ybar/kbar+1-delta)
a = np.zeros((3,3))
a[0,0]=-1/(2*cbar*cbar)*aa*aa-1/(2*cbar)*theta*(1-theta)*ybar/(kbar*kbar);
a[0,1]=1/(2*cbar*cbar)*aa;
a[1,0]=1/(2*cbar*cbar)*aa;
a[0,2]=-1/(2*cbar*cbar)*aa*(1-theta)*ybar/hbar;
a[0,2]=a[0,2]+1/(2*cbar)*theta*(1-theta)*ybar/(kbar*hbar);
a[2,0]=a[0,2];
a[1,1]=-1/(2*cbar*cbar);
a[1,2]=1/(2*cbar*cbar)*(1-theta)*ybar/hbar;
a[2,1]=a[1,2];
a[2,2]=-1/(2*cbar*cbar)*(1-theta)*ybar/hbar*(1-theta)*ybar/hbar;
a[2,2]=a[2,2]-1/(2*cbar)*theta*(1-theta)*ybar/(hbar*hbar);
a[2,2]=a[2,2]-A/(2*(1-hbar)*(1-hbar));
x=np.transpose((kbar, kbar, hbar));
m = np.zeros((4,4))
m[0,0]=math.log(kbar**theta*hbar**(1-theta)-delta*kbar)+A*math.log(1-hbar);
mm1=1/cbar*(theta*ybar/kbar+1-delta);
mm2=(1-theta)*ybar/(cbar*hbar)-A/(1-hbar);
m[0,0]=m[0,0]-mm1*kbar+kbar/cbar-mm2*hbar;
m[0,0]=m[0,0]+np.dot(np.dot(np.transpose(x),a),x)
m[0,1]=mm1/2-1*np.dot(np.transpose(a[0:3,0]),x)
m[1,0]=m[0,1]
m[0,2]=-1/(2*cbar)-1*np.dot(np.transpose(a[0:3,1]),x)
m[2,0]=m[0,2]
m[0,3]=mm2/2-1*np.dot(np.transpose(a[0:3,2]),x)
m[3,0]=m[0,3]
m[1:4,1:4]=a
AA=((1, 0),
    (0, 0))
B=((0, 0),
    (1, 0))
R=m[0:2,0:2]
Q=m[2:4,2:4]
W=np.transpose(m[0:2,2:4])
P=((1, 0),
    (0, 1))
%iterating the Ricotti equation
for i in range(1,1000):
    zinv=np.linalg.inv(Q+beta*np.dot(np.dot(np.transpose(B),P),B))
    z2=beta*np.dot(np.dot(np.transpose(AA),P),B)+np.transpose(W)
    P=R+beta*np.dot(np.dot(np.transpose(AA),P),AA)-np.dot(np.dot(z2,zinv),np.transpose(z2))


%finding the policy function
P=P
F=-np.dot(zinv,W+beta*np.dot(np.dot(np.transpose(B),P),AA))





