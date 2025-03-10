from math import *
import numpy as np

# Geometric parameters
l = 0.09
d = 0.118
# Dynamic parameters
ZZ1R = 1.0 * 0.045 * 0.045
ZZ2R = 1.0 * 0.045 * 0.045
mR = 0.5


def dgm(q11, q21, assembly_mode):
    x = 0
    y = 0
    A22_H = 1/2*np.array([-l*cos(q21)-d+l*cos(q11),-l*sin(q21)+l*sin(q11)])
    # assembly_mode is gamma
    #a = sqrt(A22_H(1)*A22_H(1)+A22_H(2)*A22_H(2))
    a = np.linalg.norm(A22_H)
    h = sqrt(l*l-a*a)
    H_A13 = assembly_mode * h/a * np.array([[0,-1],[1,0]]) @ A22_H
    O_A13 = np.array([d/2,0]) + np.array([l*cos(q21),l*sin(q21)]) + A22_H + H_A13
    x = O_A13[0]
    y = O_A13[1]
    return x, y


def igm(x, y, gamma1, gamma2):
    q11 = 0
    q21 = 0
    O_A13 = np.array([x,y])
    # left arm
    O_A11 = np.array([d/2,0])
    A11_A13 = O_A13 + O_A11
    A11_M1 = 1/2 * A11_A13
    c = np.linalg.norm(A11_M1)
    b = sqrt(l*l - c*c)
    M1_A12 = gamma1 * b/c * np.array([[0,-1],[1,0]]) @ A11_M1
    A11_A12 = A11_M1 + M1_A12
    q11 = atan2(A11_A12[1], A11_A12[0])
    # right arm
    O_A21 = np.array([-d/2,0])
    A21_A13 = O_A13 + O_A21
    A21_M2 = 1/2 * A21_A13
    c = np.linalg.norm(A21_M2)
    b = sqrt(l*l - c*c)
    M2_A22 = gamma2 * b/c * np.array([[0,-1],[1,0]]) @ A21_M2
    A21_A22 = A21_M2 + M2_A22
    q21 = atan2(A21_A22[1], A21_A22[0])
    
    
    return q11, q21


def dgm_passive(q11, q21, assembly_mode):
    q12 = 0.0
    q22 = 0.0
    x,y = dgm(q11,q21,assembly_mode)
    q12 = atan2(y/l - sin(q11), x/l + d/(2*l) - cos(q11)) - q11
    q22 = atan2(y/l - sin(q21), x/l - d/(2*l) - cos(q21)) - q21
    return q12, q22


# You can create intermediate functions to avoid redundant code
def compute_A_B(q11, q12, q21, q22):
    A = 0
    B = 0
    
    # left arm
    ux_12 = cos(q12+q11)
    uy_12 = sin(q12+q11)
    u_12 = np.array([ux_12,uy_12])
    
    ux_11 = cos(q11)
    uy_11 = sin(q11)
    u_11 = np.array([ux_11,uy_11])
    v_11 = np.array([[0,-1],[1,0]]) @ u_11
    
    # right arm
    ux_22 = cos(q22+q21)
    uy_22 = sin(q22+q21)
    u_22 = np.array([ux_22,uy_22])
    
    ux_21 = cos(q21)
    uy_21 = sin(q21)
    u_21 = np.array([ux_21,uy_21])
    v_21 = np.array([[0,-1],[1,0]]) @ u_21
    
    A = np.array([u_12,u_22])
    
    B = np.array([[l*np.dot(u_12,v_11),0],[0,l*np.dot(u_22,v_21)]])
    
    return A, B


def dkm(q11, q12, q21, q22, q11D, q21D):
	# vector x0 è il versore lungo x (1,0)
    xD = 0
    yD = 0
    # u è il versore della q, quindi prendo q e faccio ux = cos(q) e uy = sin q 
    # abbiamo in input tutte le q quindi ok, e poi per la v, facciamo q meno 90 gradi
    # e poi facciamo stessa cosa per vx e vy con cos e sin, a partire dal vettore v calcolato con q-90
    
    # dobbiamo partire direttamente dall equazione 12
    
    A, B = compute_A_B(q11, q12, q21, q22);
    
    csiD = np.linalg.inv(A) @ B @ np.array([q11D,q21D])
    
    xD = csiD[0]
    yD = csiD[1]
    
    return xD, yD


def ikm(q11, q12, q21, q22, xD, yD):
    q11D = 0
    q21D = 0
    
    A, B = compute_A_B(q11, q12, q21, q22);
    
    qD = np.linalg.inv(B) @ A @ np.array([[xD],[yD]])
    
    q11D = qD[0]
    q21D = qD[1]
    
    return q11D, q21D

def dkm_passive(q11, q12, q21, q22, q11D, q21D, xD, yD):
    q12D = 0
    q22D = 0
    
    ux_11 = cos(q11)
    uy_11 = sin(q11)
    u_11 = np.array([ux_11,uy_11])
    v_11 = np.array([[0,-1],[1,0]]) @ u_11
    
    ux_21 = cos(q21)
    uy_21 = sin(q21)
    u_21 = np.array([ux_21,uy_21])
    v_21 = np.array([[0,-1],[1,0]]) @ u_21
    
    A, B = compute_A_B(q11, q12, q21, q22);
    
    v_12 = np.array([[0,-1],[1,0]]) @ np.transpose(A[0,:])
    v_22 = np.array([[0,-1],[1,0]]) @ np.transpose(A[1,:])
    
    A = np.array([v_12,v_22])
    B = np.array([[l*np.dot(v_12,v_11)+l,0],[0,l*np.dot(v_22,v_21)+l]])
    
    q_PD = 1/l * (A @ np.array([xD,yD]) - B @ np.array([q11D,q21D]))
    q12D = q_PD[0]
    q22D = q_PD[1]
    return q12D, q22D


def dkm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD):
    xDD = 0
    yDD = 0
    A, B = compute_A_B(q11, q12, q21, q22);
    
    ux_11 = cos(q11)
    uy_11 = sin(q11)
    u_11 = np.array([ux_11,uy_11])
    v_11 = np.array([[0,-1],[1,0]]) @ u_11
    
    ux_21 = cos(q21)
    uy_21 = sin(q21)
    u_21 = np.array([ux_21,uy_21])
    v_21 = np.array([[0,-1],[1,0]]) @ u_21
    
    u_12 = np.transpose(A[0,:])
    u_22 = np.transpose(A[1,:])
    
    d = np.array([-l*q11D*q11D*np.dot(u_12,u_11)-l*(q11D+q12D)*(q11D+q12D),-l*q21D*q21D*np.dot(u_22,u_21)-l*(q21D+q22D)*(q21D+q22D)])
    
    qDD = np.array([q11DD,q21DD])
    csiDD = np.linalg.inv(A) @ (B @ qDD + d)
    xDD = csiDD[0]
    yDD = csiDD[1]
    
    return xDD, yDD


def ikm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, xDD, yDD):
    q11DD = 0
    q21DD = 0
    
    A, B = compute_A_B(q11, q12, q21, q22);
    
    ux_11 = cos(q11)
    uy_11 = sin(q11)
    u_11 = np.array([ux_11,uy_11])
    
    ux_21 = cos(q21)
    uy_21 = sin(q21)
    u_21 = np.array([ux_21,uy_21])
    
    u_12 = np.transpose(A[0,:])
    u_22 = np.transpose(A[1,:])
    
    d = np.array([-l*q11D*q11D*np.dot(u_12,u_11)-l*(q11D+q12D)*(q11D+q12D),-l*q21D*q21D*np.dot(u_22,u_21)-l*(q21D+q22D)*(q21D+q22D)])
    
    csiDD = np.array([xDD,yDD])
    
    
    qDD = np.linalg.inv(B) @ (A @ csiDD - d)
    q11DD = qDD[0]
    q21DD = qDD[1]
    
    
    return q11DD, q21DD


def dkm2_passive(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD, xDD, yDD):
    q12DD = 0
    q22DD = 0
    
    ux_11 = cos(q11)
    uy_11 = sin(q11)
    u_11 = np.array([ux_11,uy_11])
    v_11 = np.array([[0,-1],[1,0]]) @ u_11
    
    ux_21 = cos(q21)
    uy_21 = sin(q21)
    u_21 = np.array([ux_21,uy_21])
    v_21 = np.array([[0,-1],[1,0]]) @ u_21
    
    A, B = compute_A_B(q11, q12, q21, q22);
    
    v_12 = np.array([[0,-1],[1,0]]) @ np.transpose(A[0,:])
    v_22 = np.array([[0,-1],[1,0]]) @ np.transpose(A[1,:])
    
    A = np.array([v_12,v_22])
    B = np.array([[l*np.dot(v_12,v_11)+l,0],[0,l*np.dot(v_22,v_21)+l]])
    
    d = np.array([-l*q11D*q11D*np.dot(v_12,u_11),-l*q21D*q21D*np.dot(v_22,u_21)])
    q_PDD = 1/l * (A @ np.array([xDD,yDD]) - B @ np.array([q11DD,q21DD]) - d)
    q12DD = q_PDD[0]
    q22DD = q_PDD[1]
    
    return q12DD, q22DD


def dynamic_model(q11, q12, q21, q22, q11D, q12D, q21D, q22D):
    M = np.zeros((2,2))
    c = 0
    
    A, B = compute_A_B(q11, q12, q21, q22)
    
    ux_11 = cos(q11)
    uy_11 = sin(q11)
    u_11 = np.array([ux_11,uy_11])
    
    ux_21 = cos(q21)
    uy_21 = sin(q21)
    u_21 = np.array([ux_21,uy_21])
    
    u_12 = np.transpose(A[0,:])
    u_22 = np.transpose(A[1,:])
    
    Z = np.array([[ZZ1R,0],[0,ZZ2R]])
    
    d = np.array([-l*q11D*q11D*np.dot(u_12,u_11)-l*(q11D+q12D)*(q11D+q12D),-l*q21D*q21D*np.dot(u_22,u_21)-l*(q21D+q22D)*(q21D+q22D)])
    
    J = np.linalg.inv(A) @ B
    
    M = Z + mR * np.transpose(J) @ J
    
    c = mR * np.transpose(J) @ np.linalg.inv(A) @ d
    
    return M, c
