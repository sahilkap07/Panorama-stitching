# -*- coding: utf-8 -*-
"""
Panaroma

@author: sahil
"""
import argparse
import random
import copy
import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import sys
   # you could modify this line

try: 
    folder = sys.argv[1]
except:
    folder = r"C:\Users\sahil\OneDrive\Documents\Spring\CVIP\Project\Project2\CSE473573Project_2_Sample_Test\data"


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_AREA)  #Resize
    if not img.dtype == np.uint8:
        pass
    if show:
        show_image(img)
    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()
    
def match_points(des1, des2):
    """Matches corresponding points.
    """
    matches = []
    print('matching points')
    for i in range(np.size(des1,0)):
        dist = np.sqrt(np.sum((np.subtract(des1[i], des2[0]))**2)) 
        dist_best = dist
        dist_sec = dist
        dist_best_vec = [i, 0]
        dist_sec_vec = [i, 0]
        for j in range(1, np.size(des2,0)):
            dist = np.sqrt(np.sum((np.subtract(des1[i], des2[j]))**2)) 
            if dist < dist_best:
                dist_sec = dist_best
                dist_best = dist
                dist_best_vec = [i, j]
            elif dist < dist_sec:
                dist_sec = dist
                dist_sec_vec = [i, j]
        if dist_best/dist_sec <= 0.70:
            matches.append(dist_best_vec)
    return matches

def RANSAC(x_coori, y_coori, x_coorj, y_coorj, img1, img2):
    """removes outliers using homography equation. 
    """
    print('in ransac')
    Homo=[]
    count_max = 0
    #print(len(x_coorj))
    #print(x_coori[0])
    for N in range(600):
        i = random.sample(range(len(x_coori)), 8)
        A = np.array([[x_coori[i[0]], y_coori[i[0]], 1, 0, 0, 0, -x_coorj[i[0]]*x_coori[i[0]], -x_coorj[i[0]]*y_coori[i[0]], -x_coorj[i[0]]]])
        A = np.append(A, [[0, 0, 0, x_coori[i[0]], y_coori[i[0]], 1, -y_coorj[i[0]]*x_coori[i[0]], -y_coorj[i[0]]*y_coori[i[0]], -y_coorj[i[0]]]], axis = 0)
        for pt in range(1, 8):
            A = np.append(A, [[x_coori[i[pt]], y_coori[i[pt]], 1, 0, 0, 0, -x_coorj[i[pt]]*x_coori[i[pt]], -x_coorj[i[pt]]*y_coori[i[pt]], -x_coorj[i[pt]]]], axis = 0)
            A = np.append(A, [[0, 0, 0, x_coori[i[pt]], y_coori[i[pt]], 1, -y_coorj[i[pt]]*x_coori[i[pt]], -y_coorj[i[pt]]*y_coori[i[pt]], -y_coorj[i[pt]]]], axis = 0)
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        H = vh[-1]
        count = 0
        refined_matches = []
        search_in = np.setdiff1d([*range(len(x_coori))],(i))
        for j in search_in:
            val = np.dot([[x_coori[j], y_coori[j], 1, 0, 0, 0, -x_coorj[j]*x_coori[j], -x_coorj[j]*y_coori[j], -x_coorj[j]],\
                      [0, 0, 0, x_coori[j], y_coori[j], 1, -y_coorj[j]*x_coori[j], -y_coorj[j]*y_coori[j], -y_coorj[j]]], H)
            val = np.sqrt(np.dot(np.transpose(val), val))
            if val <= 0.0095:
                count+=1
                refined_matches.append([x_coori[j], y_coori[j], x_coorj[j], y_coorj[j]])
                
        if count > count_max:
            count_max = count
            final_matches = refined_matches
            Homo = [H, img1, img2]
    print(Homo)
    print('out of ransac')
    return (Homo)
    
    
def main():

    images = []
    pts = []
    #reading the images along with the keypoints and descriptors
    for fm in os.listdir(folder):
        
        if fm != 'panorama.jpg':
            img = read_image(os.path.join(folder, fm))
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(img, None)
            im = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            images.append([img, kp, des])
            pts_ = ([[p.pt[0], p.pt[1]] for p in kp])
            pts.append(pts_)
            show_image(im)
   
    #match point calculation and storing the image pairs based on matches
    img_match = []
    list = []
    for k in range(len(images)-1):
        im_match = []
        for l in range(k+1, len(images)):
            matches = match_points(images[k][2], images[l][2])
            num_matches = len(matches)
#            if num_matches > 0.95 * np.size(images[k][2],0):
#                list = list.append(l)
            if num_matches > 0.04 * np.size(images[k][2],0) and num_matches < 0.95 * np.size(images[k][2],0):
                print(num_matches/ np.size(images[k][2],0))
                im_match.append([k, l, matches])
        im_match_ = im_match
#        if len(im_match) > 1:
#            for m in range(len(im_match)-1):
#                for n in range(len(im_match)-1, m, -1):
#                    if len((list(np.setdiff1d(im_match[m][2],im_match[n][2])))) < 0.10 * len(im_match[m][2]):
#                        im_match_.pop(n)
        img_match.append(im_match_)
    
    
       
    #taking out the coordinates of the matches pairs
    img_coor = [] 
    
    for q in range(np.size(img_match,0)):
        im_coor = []
        for r in range(np.size(img_match[q],0)): 
            cols1 = [] 
            rows1 = []
            cols2 = [] 
            rows2 = []
            for s in range(np.size(img_match[q][r][2],0)):
                cols1.append(pts[img_match[q][r][0]][img_match[q][r][2][s][0]][0])
                rows1.append(pts[img_match[q][r][0]][img_match[q][r][2][s][0]][1])
                cols2.append(pts[img_match[q][r][1]][img_match[q][r][2][s][1]][0])
                rows2.append(pts[img_match[q][r][1]][img_match[q][r][2][s][1]][1])
            im_coor.append([cols1, rows1, cols2, rows2, img_match[q][r][0], img_match[q][r][1]])
        img_coor.append(im_coor)
          

    #calculating homography using ransac and storing right pairs of images and homography 
    
    proj_match = []
    for t in range(len(img_coor)):
        pair_match = []
        for u in range(len(img_coor[t])):
            if np.sqrt(sum((np.asarray(img_coor[t][u][0]))**2))-np.sqrt(sum((np.asarray(img_coor[t][u][2]))**2)) > np.sqrt(sum((np.asarray(img_coor[t][u][1]))**2))-np.sqrt(sum((np.asarray(img_coor[t][u][3]))**2)):
                if np.sqrt(sum((np.asarray(img_coor[t][u][0]))**2)) < np.sqrt(sum((np.asarray(img_coor[t][u][2]))**2)):
                    matches = RANSAC(img_coor[t][u][0], img_coor[t][u][1], img_coor[t][u][2], img_coor[t][u][3], img_coor[t][u][5], img_coor[t][u][4])
                elif np.sqrt(sum((np.asarray(img_coor[t][u][0]))**2)) > np.sqrt(sum((np.asarray(img_coor[t][u][2]))**2)):
                    matches = RANSAC(img_coor[t][u][2], img_coor[t][u][3], img_coor[t][u][0], img_coor[t][u][1], img_coor[t][u][4], img_coor[t][u][5])
            else:
                if np.sqrt(sum((np.asarray(img_coor[t][u][1]))**2)) < np.sqrt(sum((np.asarray(img_coor[t][u][3]))**2)):
                    matches = RANSAC(img_coor[t][u][0], img_coor[t][u][1], img_coor[t][u][2], img_coor[t][u][3], img_coor[t][u][5], img_coor[t][u][4])
                elif np.sqrt(sum((np.asarray(img_coor[t][u][1]))**2)) > np.sqrt(sum((np.asarray(img_coor[t][u][3]))**2)):
                    matches = RANSAC(img_coor[t][u][2], img_coor[t][u][3], img_coor[t][u][0], img_coor[t][u][1], img_coor[t][u][4], img_coor[t][u][5])
            if len(matches) > 0:
                pair_match.append(matches)
        if len(pair_match) > 0:
            proj_match.append(pair_match) 
        
    
    #Storing images and homography matrices in right sequence 
    
    print((proj_match))
    list_of_order = []
    list_of_homo = []
    list_of_order.append(proj_match[0][0][1])
    list_of_order.append(proj_match[0][0][2])
    list_of_homo.append(np.reshape(proj_match[0][0][0],(3,3)))
    for i in range(len(proj_match)):
        for j in range(len(proj_match[i])):
                if list_of_order[-1] == proj_match[i][j][1]:
                    list_of_order.append(proj_match[i][j][2])
                    list_of_homo.append(np.reshape(proj_match[i][j][0],(3,3)))
                elif list_of_order[0] == proj_match[i][j][2]:
                    list_of_order.insert(0, proj_match[i][j][1])
                    list_of_homo.insert(0, np.reshape(proj_match[i][j][0],(3,3)))


    print(list_of_order)
    print(list_of_homo)
        
    img_list = list_of_order.copy()
    homo_list = list_of_homo.copy()
    dst2 = images[img_list[len(img_list)-1]][0]
        
    # plotting the images using matplotlib
    
    for ims in range(len(homo_list)-1, -1, -1):
        try: 
            H_new = Homo_next
        except:
            H_new = homo_list[ims]
        w = np.dot(H_new,np.array([0, 0, 1]))
        wd = np.dot(H_new,np.array([[dst2.shape[1]], [dst2.shape[0]], [1]]))
        w1 = int(abs(w[0]/w[2] - wd[0]/wd[2]))
        w2 = int(abs((w[1]/w[2]) - (wd[1]/wd[2])))
        dst2 = cv2.warpPerspective(dst2, H_new, (images[img_list[ims]][0].shape[1]  + dst2.shape[1] , dst2.shape[0]))
        #dst2 = cv2.warpPerspective(dst2, H_new, (images[img_list[ims]][0].shape[1]  + w1 , max(w2,images[img_list[ims]][0].shape[0]))
        dst2[0:images[img_list[ims]][0].shape[0], 0:images[img_list[ims]][0].shape[1]] = images[img_list[ims]][0]
        w = np.dot(H_new,np.array([0, 0, 1]))
        wd = np.dot(H_new,np.array([[dst2.shape[1]], [dst2.shape[0]], [1]]))
        if ims == 500:
            images_1 = [images[img_list[ims-1]][0], dst2]
            images_2 = []
            for ii in range(2):
                sift = cv2.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(images_1[ii], None)
                #im = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                images_2.append([ii, kp, des])
            pts_1 = ([[p.pt[0], p.pt[1]] for p in images_2[0][1]])
            pts_2 = ([[p.pt[0], p.pt[1]] for p in images_2[1][1]])
            matches = match_points(images_2[0][2], images_2[1][2])
            cols1 = [] 
            rows1 = []
            cols2 = [] 
            rows2 = []
            print(matches)
            for s in range(len(matches)):
                cols1.append(pts_1[matches[s][0]][0])
                rows1.append(pts_1[matches[s][0]][1])
                cols2.append(pts_2[matches[s][1]][0])
                rows2.append(pts_2[matches[s][1]][1])
            matches = RANSAC(cols2, rows2, cols1, rows1, 0, 1)
            Homo_next = np.reshape(matches[0],(3,3))
        plt.figure(figsize=(16,14)) 
        plt.title('Warped Image')
        plt.imshow(dst2)
        plt.show()
    #print(type(dst))
    img = dst2


    #img = img[0:int(abs(wd1)-abs(w1)), 0:int(abs(wd2)-abs(w2))]
    
    img = np.asarray(img, dtype=np.uint8)
    cv2.imwrite(os.path.join(folder , 'panorama.jpg'), img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()


#np.linalg.inv



























































































































































