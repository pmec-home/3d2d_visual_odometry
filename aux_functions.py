import numpy as np

def gen_img_txt(num_img):
    
    img_str = str(num_img)

    while len(img_str) < 6:
    
        img_str = '0' + img_str
    
    return img_str + '.png'



def linear_triangulation(u0, P0, u1, P1):
    
    X = []
    
    p1_0 = P0[0]
    p2_0 = P0[1]
    p3_0 = P0[2]
    
    p1_1 = P1[0]
    p2_1 = P1[1]
    p3_1 = P1[2]
    
    A = np.zeros((4,4))
    
    for i in range (0,np.size(u0,1)):
    
        A[0,:] = u0[0,i] * p3_0 - p1_0
        A[1,:] = u0[1,i] * p3_0 - p2_0
        A[2,:] = u1[0,i] * p3_1 - p1_1
        A[3,:] = u1[1,i] * p3_1 - p2_1
        
        U, S, V = np.linalg.svd(A)
        
        # if(V[-1][0] < 0):
        #     V = -V
        
        X.append(V[-1])
    
    return np.array(X).T

