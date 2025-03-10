from math import *
import numpy as np

# Geometric parameters
l = 0.2828427
d = 0.4
# Dynamic parameters
mp = 3.0 # end effector mass
mf = 1.0 # foot mass (the upper links are considered massless)


def dgm(q11, q21, assembly_mode):
	#print(q11-q21)
	A2_H = 1/2*np.array([-d,q11-q21])
	A2_H = A2_H.reshape(-1,1)
	# assembly_mode is gamma
	a = np.linalg.norm(A2_H)
	h = sqrt(l*l-a*a)
	E = np.array([[0,-1],[1,0]])
	H_C = assembly_mode * h/a * np.matmul(E,A2_H)
	O_C =  np.array([[d/2],[q21]]) + A2_H + H_C
	#print(O_C)
	x = O_C[0,0]
	y = O_C[1,0]
	#print(x,y)
	return x, y


def igm(x, y, gamma1, gamma2):
	q11 =  y + gamma1*sqrt(pow(l,2)-pow(x+d/2,2))
	q21 =  y + gamma1*sqrt(pow(l,2)-pow(x-d/2,2))
	return q11, q21


def dgm_passive(q11, q21, assembly_mode):
	#print(q11, q21)
	x,y = dgm(q11,q21,assembly_mode)
	q12 = atan2((y-q11),(x+(d/2)))
	q22 = atan2((y-q21),(x-(d/2)))
	return q12, q22
	
# You can create intermediate functions to avoid redundant code
def compute_A_B(q11, q12, q21, q22):
    A = 0
    B = 0
    # left arm
    ux_12 = cos(q12)
    uy_12 = sin(q12)
    u_12 = np.array([ux_12,uy_12])
    u_12 = u_12.reshape(-1,1) # column
    
    ux_11 = 0
    uy_11 = 1
    u_11 = np.array([ux_11,uy_11])
    u_11 = u_11.reshape(-1,1) #column
    
    #print(np.size(u_11,0)) # column vector
    # right arm
    ux_22 = cos(q22)
    uy_22 = sin(q22)
    u_22 = np.array([ux_22,uy_22])
    u_22 = u_22.reshape(-1,1) #column
    
    ux_21 = 0
    uy_21 = 1
    u_21 = np.array([ux_21,uy_21])
    u_21 = u_21.reshape(-1,1) #column
    
    A = np.vstack((np.transpose(u_12), np.transpose(u_22))) # vertical stack
    B11 = u_12[0]*u_11[0] + u_12[1]*u_11[1] 
    B22 = u_22[0]*u_21[0] + u_22[1]*u_21[1] 
    B11 = np.array([B11[0],0])
    B22 = np.array([0,B22[0]])
    B = np.array([B11,B22])   
    return A, B


def dkm(q11, q12, q21, q22, q11D, q21D):
    A, B = compute_A_B(q11, q12, q21, q22);
    qD = np.array([q11D,q21D])
    qD = qD.reshape(-1,1)
    csiD = np.linalg.inv(A) @ B @ qD
    
    xD = csiD[0]
    yD = csiD[1]
    
    return xD, yD


def ikm(q11, q12, q21, q22, xD, yD):
    q11D = 0
    q21D = 0
    #print(q11)
    A, B = compute_A_B(q11, q12, q21, q22);
    pD = np.array([xD,yD])
    pD = pD.reshape(-1,1)
    qD = np.linalg.inv(B) @ A @ pD
    
    q11D = qD[0]
    q21D = qD[1]
    
    return q11D, q21D


def dkm_passive(q11, q12, q21, q22, q11D, q21D, xD, yD):
    q12D = 0
    q22D = 0
    ux_11 = 0
    uy_11 = 1
    u_11 = np.array([ux_11,uy_11])
    
    ux_21 = 0
    uy_21 = 1
    u_21 = np.array([ux_21,uy_21])
    
    A, B = compute_A_B(q11, q12, q21, q22);
    
    v_12 = np.array([[0,-1],[1,0]]) @ np.transpose(A[0,:])
    v_22 = np.array([[0,-1],[1,0]]) @ np.transpose(A[1,:])
    
    A = np.stack([np.transpose(v_12),np.transpose(v_22)])
    B = np.array([[np.dot(v_12,u_11),0],[0,np.dot(v_22,u_21)]])
    
    Pdot = np.array([xD,yD])
    Pdot = Pdot.reshape(-1,1) #column
    qdot = np.array([q11D,q21D])
    qdot = qdot.reshape(-1,1) #column
    q_PD = 1/l * (A @ Pdot - B @ qdot)
    q12D = q_PD[0]
    q22D = q_PD[1]
    return q12D, q22D


def dkm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD):
    xDD = 0
    yDD = 0
    A, B = compute_A_B(q11, q12, q21, q22);
    
    ux_11 = 0
    uy_11 = 1
    u_11 = np.array([ux_11,uy_11])
    u_11 = u_11.reshape(-1,1) #column
    
    ux_21 = 0
    uy_21 = 1
    u_21 = np.array([ux_21,uy_21])
    u_21 = u_21.reshape(-1,1) #column
    
    d = -l*np.array([q12D*q12D,q22D*q22D])
    d = d.reshape(-1,1) #column
    
    
    qDD = np.array([q11DD,q21DD])
    qDD = qDD.reshape(-1,1) #column
    csiDD = np.linalg.inv(A) @ (B @ qDD + d)
    xDD = csiDD[0]
    yDD = csiDD[1]
    return xDD, yDD


def ikm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, xDD, yDD):
    q11DD = 0
    q21DD = 0
    A, B = compute_A_B(q11, q12, q21, q22);
    
    ux_11 = 0
    uy_11 = 1
    u_11 = np.array([ux_11,uy_11])
    u_11 = u_11.reshape(-1,1) #column
    
    ux_21 = 0
    uy_21 = 1
    u_21 = np.array([ux_21,uy_21])
    u_21 = u_21.reshape(-1,1) #column
    
    d = -l*np.array([q12D*q12D,q22D*q22D])
    d = d.reshape(-1,1) #column
    
    csiDD = np.array([xDD,yDD])
    csiDD = csiDD.reshape(-1,1) #column
    
    #print(B)
    qDD = np.linalg.inv(B) @ (A @ csiDD - d)
    q11DD = qDD[0]
    q21DD = qDD[1]
    
    return q11DD, q21DD


def dkm2_passive(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD, xDD, yDD):
    q12DD = 0
    q22DD = 0
    
    ux_11 = 0
    uy_11 = 1
    u_11 = np.array([ux_11,uy_11])    
    
    ux_21 = 0
    uy_21 = 1
    u_21 = np.array([ux_21,uy_21])
    
    A, B = compute_A_B(q11, q12, q21, q22);
    
    v_12 = np.array([[0,-1],[1,0]]) @ np.transpose(A[0,:])
    v_22 = np.array([[0,-1],[1,0]]) @ np.transpose(A[1,:])
    
    A = np.array([np.transpose(v_12),np.transpose(v_22)])
    B = np.array([[np.dot(v_12,u_11),0],[0,np.dot(v_22,u_21)]])
    
    csiDD = np.array([xDD,yDD])
    csiDD = csiDD.reshape(-1,1)
    qDD = np.array([q11DD,q21DD])
    qDD = qDD.reshape(-1,1)
    q_PDD = 1/l * (A @ csiDD - B @ qDD)
    q12DD = q_PDD[0]
    q22DD = q_PDD[1]
    
    return q12DD, q22DD


def dynamic_model(q11, q12, q21, q22, q11D, q12D, q21D, q22D):
    M = np.zeros((2, 2))
    c = 0
    
    
    A, B = compute_A_B(q11, q12, q21, q22)
    
    ux_11 = 0
    uy_11 = 1
    u_11 = np.array([ux_11,uy_11])
    
    ux_21 = 0
    uy_21 = 1
    u_21 = np.array([ux_21,uy_21])
    
    u_12 = np.transpose(A[0,:])
    u_22 = np.transpose(A[1,:])
    
    
    d = l * np.array([q12D*q12D,q22D*q22D])
    mass = np.array([[mf,0],[0,mf]])
        
    J = np.linalg.inv(A) @ B
    M = mass + mp * np.transpose(J) @ J
    c = - mp * np.transpose(J) @ np.linalg.inv(A) @ d
    
    return M, c
