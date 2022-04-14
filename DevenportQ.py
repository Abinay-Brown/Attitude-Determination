'''
q method for solving Wahba's problem
Name: Abinay Brown
Email: abrown472@gatech.edu
'''

import numpy as np
from QUEST import DCM_to_Q, quat2rot



def q_method(vi, vb, w=1):
    # vi - numpy array list of vectors in inertial frame
    # vb - numpy array list of vectors in body frame
    # w - weighting vector by default 0
    B = np.dot(vb.T, vi)
    S = B + B.T

    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]]);
    sigma = np.trace(B)
    #print(S)
    #print(np.dot(S,S))
    K11 = S - (sigma * np.eye(3))
    # Assembling the K matrix.
    K = np.array([[K11[0, 0], K11[0, 1], K11[0, 2], Z[0]], [K11[1, 0], K11[1, 1], K11[1, 2], Z[1]],
                  [K11[2, 0], K11[2, 1], K11[2, 2], Z[2]], [Z[0], Z[1], Z[2], sigma]]);
    eigval, eigvec = np.linalg.eig(K)

    # Largest eigenvalue of K matrix maximizes the gain function
    ind = eigval.tolist().index(max(eigval))
    #print(max(eigval))
    # Corresponding eigenvector is the least-squares estimate and gives the quaternion for rotation.
    quat = eigvec[0:4,ind]
    #quat[0:3] = -quat[0:3]
    #print(quat);
    # Assembling the quaternion in this form q = q0 + q1i + q2j +q3k
    quaternion = np.array([quat[3], quat[0], quat[1], quat[2]])
    print(quaternion)
    dcm = quat2rot(quaternion)

    return dcm


if __name__ == "__main__":
    '''
    # inertial frame vectors.
    # vectors stored row-wise
    v1 = np.array([0.2673, 0.5345, 0.8018])
    v2 = np.array([-0.3124, 0.9370, 0.1562])
    vi = np.vstack((v1, v2))

    # body frame vectors.
    # vectors stored row-wise
    v1 = np.array([0.7814, 0.3751, 0.4987])
    v2 = np.array([0.6163, 0.7075, -0.3459])
    vb = np.vstack((v1, v2))
    '''

    # inertial frame vectors.
    # vectors stored row-wise
    v1 = np.array([0, 0.99, 0])
    v2 = np.array([0, 1.01, 0])
    vi = np.vstack((v1, v2))

    # body frame vectors.
    # vectors stored row-wise
    v1 = np.array([0.99, 0, 0])
    v2 = np.array([1.01, 0, 0])
    vb = np.vstack((v1, v2))
    print(q_method(vi, vb))
