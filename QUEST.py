import numpy as np

def DCM_to_Q( dcm ):

    if len( dcm ) != 3:
        print( "Wrong number of input rows, output is wrong!" )

    tracy = np.trace( dcm )

    q0s = 0.25*( 1.0 + tracy )
    q1s = 0.25*( 1.0 + 2.0*dcm[ 0 ][ 0 ] - tracy )
    q2s = 0.25*( 1.0 + 2.0*dcm[ 1 ][ 1 ] - tracy )
    q3s = 0.25*( 1.0 + 2.0*dcm[ 2 ][ 2 ] - tracy )

    # Find largest q^2 value and use it for division to ensure no 0 div
    qden = max( q0s , q1s , q2s , q3s )

    if qden == q0s:
        qden = np.sqrt( qden )
        qu = [ qden ,
               0.25*( dcm[ 1 ][ 2 ] - dcm[ 2 ][ 1 ] )/qden ,
               0.25*( dcm[ 2 ][ 0 ] - dcm[ 0 ][ 2 ] )/qden ,
               0.25*( dcm[ 0 ][ 1 ] - dcm[ 1 ][ 0 ] )/qden ]
    elif qden == q1s:
        qden = np.sqrt( qden )
        qu = [ 0.25*( dcm[ 1 ][ 2 ] - dcm[ 2 ][ 1 ] )/qden ,
               qden ,
               0.25*( dcm[ 0 ][ 1 ] + dcm[ 1 ][ 0 ] )/qden ,
               0.25*( dcm[ 2 ][ 0 ] + dcm[ 0 ][ 2 ] )/qden ]
    elif qden == q2s:
        qden = np.sqrt( qden )
        qu = [ 0.25*( dcm[ 2 ][ 0 ] - dcm[ 0 ][ 2 ] )/qden ,
               0.25*( dcm[ 0 ][ 1 ] + dcm[ 1 ][ 0 ] )/qden ,
               qden ,
               0.25*( dcm[ 1 ][ 2 ] + dcm[ 2 ][ 1 ] )/qden ]
    elif qden == q3s:
        qden = np.sqrt( qden )
        qu = [ 0.25*( dcm[ 0 ][ 1 ] - dcm[ 1 ][ 0 ] )/qden ,
               0.25*( dcm[ 2 ][ 0 ] + dcm[ 0 ][ 2 ] )/qden ,
               0.25*( dcm[ 1 ][ 2 ] + dcm[ 2 ][ 1 ] )/qden ,
               qden ]
    else:
        print( "Something's wrong with your qden values - none is maximum!" )

    # Choose the short rotation (q0 > 0)
    if qu[ 0 ] <= 0.0:
        for i in range( 0 , 4 ):
            qu[ i ] = - qu[ i ]

    return qu

def QUEST(W, Vi, Vb):

    # Computing Optimal Lambda
    lambda_opt = 0;
    for i in range(0, len(W)):
        lambda_opt = lambda_opt + W[i]

    # Constructing B matrix (Attitude Profile Matrix)
    temp = np.zeros([3, 3])
    B = np.zeros([3,3])
    for i in range(0, len(W)):
        for j in range(0,3):
            for k in range(0,3):
                temp[k, j] = Vi[i, j]*Vb[i, k]
        B = B + (temp*W[i])
    print(B)
    # Method of Sequential Rotation
    R1 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    R2 = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    R3 = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    R4 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    cos_phi_min = 0.5#np.cos(np.deg2rad(120))
    B1 = np.dot(B, R1)
    B2 = np.dot(B, R2)
    B3 = np.dot(B, R3)
    B4 = np.dot(B, R4)
    flag = 0

    if np.trace(B1) >= cos_phi_min:
        B = B1
        flag = 1
    elif np.trace(B2) >= cos_phi_min:
        B = B2
        flag = 2
    elif np.trace(B3) >= cos_phi_min:
        B = B3
        flag = 3
    elif np.trace(B4) >= cos_phi_min:
        B = B4
        flag = 4
    print(B)
    print("sequence: {flg:d}".format(flg=flag))
    # Constructing S matrix
    S = B + B.T
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    sigma = 0.5 * np.trace(S)
    kappa = np.trace(adj(S))
    delta = np.linalg.det(S)
    a = sigma**2 - kappa
    b = sigma**2 + np.dot(Z.T, Z)
    c = delta + np.dot(Z.T, np.dot(S, Z))
    d = np.dot(Z.T, np.dot(S, np.dot(S, Z)))
    epsilon = 10**-15;
    lam = lambda_opt
    while 1:
        lam_old = lam;
        f = (lam**4)-((a+b) * lam**2) - (c * lam) + ((a * b)+(c * sigma)-d)
        f_prime = (4*lam**3)-(2 * (a+b) * lam) - c
        lam = lam - (f / f_prime);
        ea = np.fabs((lam -lam_old) / lam) * 100;
        if (ea <= epsilon):
            break

    aug = ((lam+sigma)*np.eye(3)) - S
    inv_aug = (adj(aug))/np.linalg.det(aug)
    # rodrigues parameter
    q = np.dot(inv_aug, Z)
    mag = 1 / np.sqrt(1 + ((q[0]**2) + (q[1]**2) + (q[2]**2)))
    qt = np.zeros([4])
    qt[1] = q[0] * mag;
    qt[2] = q[1] * mag;
    qt[3] = q[2] * mag;
    qt[0] = mag;

    dcm = Q_to_DCM(qt)

    if flag == 1:
        dcm = np.dot(dcm, R1)
    elif flag == 2:
        dcm = np.dot(dcm, R2)
    elif flag == 3:
        dcm = np.dot(dcm, R3)
    elif flag == 4:
        dcm = np.dot(dcm, R4)

    # print(qt)
    q =DCM_to_Q(dcm)
    print(q)
    return dcm


def adj(mat):
    temp = np.zeros([3, 3])
    temp[0][0] = ((mat[1][1] * mat[2][2]) - (mat[2][1] * mat[1][2]))
    temp[0][1] = -((mat[1][0] * mat[2][2]) - (mat[2][0] * mat[1][2]))
    temp[0][2] = ((mat[1][0] * mat[2][1]) - (mat[2][0] * mat[1][1]))

    temp[1][0] = -((mat[0][1] * mat[2][2]) - (mat[2][1] * mat[0][2]))
    temp[1][1] = ((mat[0][0] * mat[2][2]) - (mat[2][0] * mat[0][2]))
    temp[1][2] = -((mat[0][0] * mat[2][1]) - (mat[2][0] * mat[0][1]))

    temp[2][0] = ((mat[0][1] * mat[1][2]) - (mat[1][1] * mat[0][2]))
    temp[2][1] = -((mat[0][0] * mat[1][2]) - (mat[1][0] * mat[0][2]))
    temp[2][2] = ((mat[0][0] * mat[1][1]) - (mat[1][0] * mat[0][1]))
    return temp
def Q_to_DCM( qu ):

    if len( qu ) != 4:
        print( "Wrong Q length size, output is wrong!" )

    # Renorm quaternion to make sure it's unit before DCM
    qnorm = np.sqrt( np.dot( qu , qu ) )
    for i in range( 0 , 4 ):
        qu[ i ] /= qnorm

    # Compute squares of quaternion
    q0s = qu[ 0 ]*qu[ 0 ]
    q1s = qu[ 1 ]*qu[ 1 ]
    q2s = qu[ 2 ]*qu[ 2 ]
    q3s = qu[ 3 ]*qu[ 3 ]

    dcm = [ [ q0s + q1s - q2s - q3s , 2.0*( qu[ 1 ]*qu[ 2 ] + qu[ 0 ]*qu[ 3 ] ) , 2.0*( qu[ 1 ]*qu[ 3 ] - qu[ 0 ]*qu[ 2 ] ) ] ,
            [ 2.0*( qu[ 2 ]*qu[ 1 ] - qu[ 0 ]*qu[ 3 ] ) , q0s - q1s + q2s - q3s , 2.0*( qu[ 2 ]*qu[ 3 ] + qu[ 0 ]*qu[ 1 ] ) ] ,
            [ 2.0*( qu[ 3 ]*qu[ 1 ] + qu[ 0 ]*qu[ 2 ] ) , 2.0*( qu[ 3 ]*qu[ 2 ] - qu[ 0 ]*qu[ 1 ] ) , q0s - q1s - q2s + q3s ] ]

    return dcm

def quat2rot(q):
    # Convert quaternion to rotation matrix
    # quaternion given by q = q0 + q1i + q2j +q3k
    # where q1 q2 q3 define unit vector for the axis of rotation.
    # q0 define the rotation about the axis.
    rot11 = 2 * (q[0] ** 2 + q[1] ** 2) - 1;
    rot12 = 2 * ((q[1] * q[2]) - (q[0] * q[3]));
    rot13 = 2 * ((q[1] * q[3]) + (q[0] * q[2]));
    rot21 = 2 * ((q[1] * q[2]) + (q[0] * q[3]));
    rot22 = 2 * (q[0] ** 2 + q[2] ** 2) - 1;
    rot23 = 2 * ((q[2] * q[3]) - (q[0] * q[1]));
    rot31 = 2 * ((q[1] * q[3]) - (q[0] * q[2]));
    rot32 = 2 * ((q[2] * q[3]) + (q[0] * q[1]));
    rot33 = 2 * (q[0] ** 2 + q[3] ** 2) - 1;
    rot = np.array([[rot11, rot12, rot13], [rot21, rot22, rot23], [rot31, rot32, rot33]])
    return np.round(rot, 5);
