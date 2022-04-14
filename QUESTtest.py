import numpy as np
from QUEST import QUEST
from q_method import q_method
if __name__ == '__main__':
    # Weight matrix
    w = np.array([1, 1])
    w = w.T


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


    # inertial frame vectors.
    # vectors stored row-wise
    v1 = np.array([-1.0/np.sqrt(2),-1.0/np.sqrt(2) , 0])
    v2 = np.array([-1.0/np.sqrt(2),-1.0/np.sqrt(2) , 0])
    vi = np.vstack((v1, v2))
    print("Inertial Frame : {i:.4f}i {j:.4f}j {k:.4f}k".format(i=vi[0,0], j = vi[0,1], k = vi[0,2]))
    # body frame vectors.
    # vectors stored row-wise
    v1 = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0])
    v2 = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0])
    vb = np.vstack((v1, v2))
    print("Body Frame : {i:.4f}i {j:.4f}j {k:.4f}k".format(i=vb[0,0], j=vb[0,1], k=vb[0,2]))
    dcm = QUEST(w, vi, vb)
    string = " {a:0.3f} {b:0.3f} {c:0.3f} \n"

    for i in range(0,3):
        print(string.format(a=dcm[i][0], b=dcm[i][1], c=dcm[i][2]))

    print("---------------------------------------")
    #dcm = q_method(vi, vb)
    #print(dcm)
    print("Calculated QUEST Body Frame: {i:.4f}i {j:.4f}j {k:.4f}k".format(i=np.dot(dcm, vi[0, 0:3].T)[0], j = np.dot(dcm, vi[0, 0:3].T)[1], k = np.dot(dcm, vi[0, 0:3].T)[2]))

