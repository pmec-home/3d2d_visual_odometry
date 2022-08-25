import math
from multiprocessing.sharedctypes import RawValue
import numpy as np
import cv2 as cv

from aux_functions import *

def main():

    #Initializations

    sequence = '00'

    cam0_path = 'data/' + sequence + '/image_0/'
    cam1_path = 'data/' + sequence + '/image_1/'

    poseFile = 'data/' + sequence + '/' + sequence + '.txt'

    fpPoseFile = open(poseFile, 'r')
    groundTruthTraj = fpPoseFile.readlines()

    pose_it = 0

    cur_t = None
    cur_R = None

    traj = np.zeros((1400,1400,3), dtype=np.uint8)
    xpos = 600
    ypos = 200

    #cam0
    cam0_K = np.matrix([[718.856, 0, 607.1928],
                   [0, 718.856, 185.2157],
                   [0, 0, 1]])

    cam0_P = np.matrix([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00], 
                        [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], 
                        [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]
    ])

    #cam1
    cam1_P = np.matrix([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02], 
                        [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00], 
                        [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]
    ])

    #feature detector and matching
    fast = cv.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    step = 1

    while True:
    
        num_img_0 = gen_img_txt(pose_it)
        num_img_1 = gen_img_txt(pose_it + step)
    
        #Leitura das imagens de cam0 e cam1 no tempo t

        img00 = cv.imread(cam0_path + num_img_0, cv.IMREAD_GRAYSCALE)
        img01 = cv.imread(cam1_path + num_img_0, cv.IMREAD_GRAYSCALE)
    
        #Leitura da imagem de cam0 no tempo t+1

        img10 = cv.imread(cam0_path + num_img_1, cv.IMREAD_GRAYSCALE)

        if img10 is None:
            break
    
        #Detecção de pontos (FAST)

        kp00 = fast.detect(img00,None)
        kp01 = fast.detect(img01,None)
    
        #Cálculo dos descritores (BRIEF)

        kp00, des00 = brief.compute(img00,kp00)
        kp01, des01 = brief.compute(img01,kp01)
    
        #Matching entre as câmeras da esquerda e da direita em t e t+1

        matches00 = bf.match(des00, des01)
    
        #Extração das coordenadas dos keypoints
    
        new_kp00 = []
        new_kp01 = []
    
        for i in range(0, len(matches00)):
            if matches00[i].distance < 15:
                new_kp00.append(kp00[matches00[i].queryIdx])
                new_kp01.append(kp01[matches00[i].trainIdx])
        
        pts00 = np.int32(cv.KeyPoint_convert(new_kp00))
        pts01 = np.int32(cv.KeyPoint_convert(new_kp01))
    
        #Seleção dos melhores pontos

        best_pts00 = []
        best_pts01 = []

        for i in range(0, len(pts00)):
        
            desl00_x = range(pts00[i][0] - 80, pts00[i][0]-5)
        
            if pts01[i][0] in desl00_x:
                
                desl00_y = range(pts00[i][1]-5, pts00[i][1]+5)
            
                if pts01[i][1] in desl00_y:
                
                    best_pts00.append(pts00[i])
                    best_pts01.append(pts01[i])
    
        best_pts00 = np.float32(best_pts00)
        best_pts01 = np.float32(best_pts01)
    
        #Tracking entre dos pontos da camera esquerda em t e t+1
    
        lk_params = dict(
            winSize = (15, 15),
            maxLevel = 3,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.01)
        )
    
        pts10, st, err = cv.calcOpticalFlowPyrLK(img00, img10, best_pts00, None, flags= cv.MOTION_AFFINE, **lk_params)

        pts10 = np.around(pts10)

        best_pts00 = best_pts00[st.ravel() == 1]
        best_pts01 = best_pts01[st.ravel() == 1]
        pts10 = pts10[st.ravel() == 1]
        err = err[st.ravel() == 1]

        best_pts00 = best_pts00[err.ravel() >= 0]
        best_pts01 = best_pts01[err.ravel() >= 0]
        pts10 = pts10[err.ravel() >= 0]
        err = err[err.ravel() >= 0]

        best_pts00 = best_pts00[err.ravel() < 15]
        best_pts01 = best_pts01[err.ravel() < 15]
        pts10 = pts10[err.ravel() < 15]
    
        #Triangulação

        X0 = cv.triangulatePoints(cam0_P, cam1_P, best_pts00.T, best_pts01.T)

        X0 = X0/X0[3]
    
        #Cálculo da transformação com PnP
        
        X0 = X0[:3,:]


        X0 = np.array(X0.T, ndmin = 3)
    
        _, R, t, inliers = cv.solvePnPRansac(X0, pts10, cam0_K, None, iterationsCount=200, reprojectionError=1.0)
        
        R, _ = cv.Rodrigues(R)
    
        #Plot resultados
        
        if(pose_it > 0):
            cur_t = cur_t - cur_R.T @ t
            cur_R = R.dot(cur_R)
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
            
        else:
            x, y, z = 0, 0, 0
            cur_R = R
            cur_t = t

        draw_x, draw_y = int(x) + xpos, int(z) + ypos
        grndPose = groundTruthTraj[pose_it + step].strip().split()
        grndX = int(float(grndPose[3])) + xpos
        grndY = int(float(grndPose[11])) + ypos
        
        cv.circle(traj, (grndX,grndY), 1, (0,0,255), 2)
        cv.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        cv.circle(traj, (draw_x,draw_y), 1, (pose_it*255/4540,255-pose_it*255/4540,0), 1)
        cv.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm "%(x,y,z)
        cv.putText(traj, text, (20,40), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        cv.imshow('Camera view', img10)
        cv.imshow('Trajectory', traj)
        cv.waitKey(1)
        
        pose_it += step
    
    cv.imwrite(sequence + '.png', traj)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()