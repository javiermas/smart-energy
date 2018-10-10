#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:48:03 2018

@author: yc00059
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import traces

n=3 # number of nodes of the network
t=720*30 # number of timesteps

# Desired battery range
SOCM=0.95
SOCm=0.3

# Physical battery range
SM=0.99
Sm=0.2

# Production/Load minimum threshold
Pm=0
Lm=0

# Initialize E, Co, S, B, I
#E=np.full((n,t),0)
E=np.zeros((n,t))
Co=np.zeros((n,t))
S=np.zeros((n,t))
B=np.zeros((n,t))
Ig=np.zeros((n,t))
Bal=np.zeros((n,t))

# IMPORT P AND L
df1=pd.read_excel('SB71.xlsx')
df2=pd.read_excel('SB309.xlsx')
df3=pd.read_excel('SB32.xlsx')

n1='71'
n2='309'
n3='32'

df1=df1.set_index('TIME')                      # Set TIME as the index
df1=df1.groupby(pd.TimeGrouper('2min')).mean() # Regularize timestamping
df1=df1.fillna(method='ffill')                 # Fill blanks with the previous value

df2=df2.set_index('TIME')                      
df2=df2.groupby(pd.TimeGrouper('2min')).mean() 
df2=df2.fillna(method='ffill')                 

df3=df3.set_index('TIME')                      
df3=df3.groupby(pd.TimeGrouper('2min')).mean() 
df3=df3.fillna(method='ffill')

b=0

D1=df1.values
P1=D1[b:(b+t),0]+D1[b:(b+t),1]
L1=D1[b:(b+t),2]

a=(8640-2160+360)*1

D2=df2.values
P2=D2[a:(a+t),0]+D2[a:(a+t),1]
L2=D2[a:(a+t),2]

D3=df3.values
P3=D3[a:(a+t),0]+D3[a:(a+t),1]
L3=D3[a:(a+t),2]

P=np.divide([P1,P2,P3],30) # Convert Inputs from W to Wh
L=np.divide([L1,L2,L3],30)

##################

SOC=[0.5,0.5,0.5] # Initial SOC
    
vC=pd.Series([204,166,115,136,150,205,194,402,597,67,199,71,504,340,77],index=['32','34','37','43','47','48','68','71','97','190','309','344','345','366','369'])  
C=[vC[n1],vC[n2],vC[n3]] # Battery capacity in [Wh]

for k in range(0,t):
    
    if k>0:
        for j in range(0,n):
            S[j][k]=S[j][k-1]
    else:
        for j in range(0,n):
            S[j][k]=SOC[j]
    
    for i in range(0,n):
        
        if P[i][k]>Pm and S[i][k]<SOCM: # Charge local battery
            S[i][k]=S[i][k]+(P[i][k]/C[i])
            B[i][k]=B[i][k]+P[i][k]
            E[i][k]=0
        else:
            S[i][k]=S[i][k]
            E[i][k]=P[i][k]
            
        if L[i][k]>Lm and S[i][k]>SOCm: # Load supplied from local battery
            S[i][k]=S[i][k]-L[i][k]/C[i]
            B[i][k]=B[i][k]-L[i][k]
            Co[i][k]=0
        else:
            S[i][k]=S[i][k]
            Co[i][k]=L[i][k]
        
        for j in range(0,n): # Check SOC is between SOCM and SOCm
            if S[j][k]<Sm:
                B[j][k]=B[j][k]-(S[j][k]-Sm)*C[j]
                Co[j][k]=(Sm-S[j][k])*C[j]
                S[j][k]=Sm
                
            elif S[j][k]>SM:
                B[j][k]=B[j][k]-(S[j][k]-SM)*C[j]
                E[j][k]=(S[j][k]-SM)*C[j]
                S[j][k]=SM

x=np.array(range(1,t+1))

Curt=100*E/P
Shed=100*Co/L

for k in range(0,t):
    for i in range(0,n):
        Bal[i,k]=(P[i,k]-E[i,k])-(L[i,k]-Co[i,k])-B[i,k]-Ig[i,k]

plt.figure()
plt.title('Production')
plt.plot(x,P[0,:],x,P[1,:],x,P[2,:])
plt.figure()
plt.title('Load')
plt.plot(x,L[0,:],x,L[1,:],x,L[2,:])
plt.figure()
plt.title('Battery')
plt.plot(x,B[0,:],x,B[1,:],x,B[2,:])
plt.figure()
plt.title('SOC')
plt.plot(x,S[0,:],x,S[1,:],x,S[2,:])
plt.figure()
plt.title('Balance')
plt.plot(x,Bal[0,:],x,Bal[1,:],x,Bal[2,:])
plt.figure()
plt.title('PV curtailment')
plt.plot(x,Curt[0,:],x,Curt[1,:],x,Curt[2,:])
plt.figure()
plt.title('Load shedding')
plt.plot(x,Shed[0,:],x,Shed[1,:],x,Shed[2,:])

print('TOTAL CURTAILMENT of NODE 0=',100*sum(E[0,:])/sum(P[0,:]),'%')
print('TOTAL CURTAILMENT of NODE 1=',100*sum(E[1,:])/sum(P[1,:]),'%')
print('TOTAL CURTAILMENT of NODE 2=',100*sum(E[2,:])/sum(P[2,:]),'%')
print('AVERAGE CURTAILMENT=',100*(sum(E[0,:])+sum(E[1,:])+sum(E[2,:]))/(sum(P[0,:])+sum(P[1,:])+sum(P[2,:])),'%')

print('TOTAL LOAD SHEDDING of NODE 0=',100*abs(sum(Co[0,:]))/sum(L[0,:]),'%')
print('TOTAL LOAD SHEDDING of NODE 1=',100*abs(sum(Co[1,:]))/sum(L[1,:]),'%')
print('TOTAL LOAD SHEDDING of NODE 2=',100*abs(sum(Co[2,:]))/sum(L[2,:]),'%')
print('AVERAGE SHEDDING=',100*(sum(Co[0,:])+sum(Co[1,:])+sum(Co[2,:]))/(sum(L[0,:])+sum(L[1,:])+sum(L[2,:])),'%')