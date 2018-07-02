#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYNTHIA --> KITTI data format converter

Created on Wed Jun 13 13:27:53 2018

@author: sohom31
"""

# Imports

import json
import shutil
import cv2 as cv
import numpy as np
from numpy.linalg import inv


# -----------------------------------------------------------------------------

def load_lidar_points(pc_path, info_path):
    # Read json
    pc_data = json.load(open(pc_path))
    data = json.load(open(info_path))
    
    # Load lidar points in world coordinates
    
    pc_list = pc_data["PointCloud"]["pointList"]

    instance_ids = pc_data["BBoxIds"]["idList"]
    
    points = np.zeros((len(pc_list), 3))
    
    x_lidar = np.asarray([pc_entry["lidarPosition"]["x"] for pc_entry in pc_list])
    y_lidar = np.asarray([pc_entry["lidarPosition"]["y"] for pc_entry in pc_list])
    z_lidar = np.asarray([pc_entry["lidarPosition"]["z"] for pc_entry in pc_list])
    
    x_local = np.asarray([pc_entry["pointLocalPosition"]["x"] for pc_entry in pc_list])
    y_local = np.asarray([pc_entry["pointLocalPosition"]["y"] for pc_entry in pc_list])
    z_local = np.asarray([pc_entry["pointLocalPosition"]["z"] for pc_entry in pc_list])
    
    x_global = x_lidar + x_local
    y_global = y_lidar + y_local
    z_global = z_lidar + z_local
    
    points[:, 0] = x_global
    points[:, 1] = y_global
    points[:, 2] = z_global
        
    # Transform the 3d world --> 3d camera points
    
    points_homg_world = np.ones((len(pc_list), 4))
    points_homg_world[:, :3] = points
    
    points_homg_world = np.transpose(points_homg_world)
    
    M = np.reshape(data['extrinsic']['matrix'], [4, 4])
    
    points_homg_camera = np.matmul(inv(M), points_homg_world)
    points_homg_camera = np.transpose(points_homg_camera)
    
    points_homg_camera_kitti = np.ones((len(pc_list), 4))
    points_homg_camera_kitti[:, 0] = points_homg_camera[:, 2]
    points_homg_camera_kitti[:, 1] = -points_homg_camera[:, 0]
    points_homg_camera_kitti[:, 2] = points_homg_camera[:, 1]
    points_homg_camera_kitti[:, 3] = 0.3
    

    

    
    # Set the fourth column (reflectance) to 0.3
    
    #points_homg_camera[:, 3] = 0.3
    
    
    return points_homg_camera_kitti, instance_ids
    

# -----------------------------------------------------------------------------

def load_calib(info_path):
    # Read json
    data = json.load(open(info_path))
    
    # Load camera parameters
    Q = np.reshape(data['intrinsic']['matrix'], [4, 4])
    
    K = [[Q[2, 3], 0, Q[0, 3]],[0, Q[2, 3], Q[1, 3]],[0, 0, 1]]
    K = np.asarray(K, dtype=np.float32)
    K34 = np.zeros((3,4))
    K34[:3, :3] = K
    
    # Calculate the KITTI camera parameters
    R = np.eye(3, 3) 
    velo_to_cam = np.array([[0.0,-1.0,0.0,0.0], [0.0,0.0,-1.0,0.0], [1.0,0.0,0.0,0.0]])
    P = K34
    
    # NEW
    P[0,2] = -P[0,2]
    P[1,2] = -P[1,2]
    
    R = np.reshape(R, (1, 9))
    velo_to_cam = np.reshape(velo_to_cam, (1, 12))
    P = np.reshape(P, (1, 12))
    
    return P, R, velo_to_cam

# -----------------------------------------------------------------------------
    
def instance_label_arr_generator(info_path, sem_path):
        
    # Get the lables for the required instances in following format:
    # instance_ids --> label --> sequence id (out of 1 - 316)
    
    # Read json
    data = json.load(open(info_path))

    v_bboxes = data['bounds']['objectList'] # len --> number of instances
    
    # -------------------------------------------------------------------------
    
    # New part for Semantic Image for removing occlusions
    
    semgInst_img = cv.imread(sem_path)
    semgInst_img = semgInst_img[..., ::-1]
    
    # Read semantic image
    # segm_img = semgInst_img[:,:,0]
    
    # Read instance image
    inst_img = 255.0 * semgInst_img[:,:, 1] + semgInst_img[:,:, 2]
    inst_img_arr = np.reshape(inst_img, [inst_img.shape[0]*inst_img.shape[1], 1])
    
    # -------------------------------------------------------------------------

    instance_label_arr = np.zeros((len(instance_ids), 3), dtype = float)
    
    j = 0
    carped_class_ids = [12, 14]
    
    for i in range(0, len(v_bboxes)):
        idx =  data['bounds']['objectList'][i]["id"]
        label = data['bounds']['objectList'][i]["label"]
        static_flag = data['bounds']['objectList'][i]["isStatic"]
        seq_no = i
        
        if idx in instance_ids and idx in inst_img_arr and (label in carped_class_ids or (label == 19 and static_flag == False)):
            instance_label_arr[j, 0] = idx
            instance_label_arr[j, 1] = label
            instance_label_arr[j, 2] = seq_no
            j = j + 1
            
    instance_label_arr = instance_label_arr[0:j, :] # include nonzero entries only
    
    return instance_label_arr
    
# -----------------------------------------------------------------------------
    
def labelarr_generator(info_path, image_path, instance_label_arr):
    
    # Read json
    data = json.load(open(info_path))
    v_bboxes = data['bounds']['objectList'] # len --> number of instances
    
    # -----------------------------------------------------------------------------
    
    # Dimensions for RGB images
    width = 1242
    height = 375
    
    img = cv.imread(image_path)
    dummy_img = img
    
    img = img[...,::-1]
    
    img_size = img.shape
    
    # Read 3d bboxes
    
    v_bboxes = data['bounds']['objectList'] # len --> number of instances
    
    # [number of instances * 8, 4] 
    
    points3d_world = np.zeros((len(v_bboxes)*8,4), dtype=np.float32)
    
    for j in range(len(v_bboxes)):
        points3d_world[j*8,:] = [float(v_bboxes[j]['FrontTopLeft']['x']), float(v_bboxes[j]['FrontTopLeft']['y']), float(v_bboxes[j]['FrontTopLeft']['z']), 1.]
        points3d_world[j*8+1,:] = [float(v_bboxes[j]['FrontTopRight']['x']), float(v_bboxes[j]['FrontTopRight']['y']), float(v_bboxes[j]['FrontTopRight']['z']), 1.]
        points3d_world[j*8+2,:] = [float(v_bboxes[j]['FrontBottomLeft']['x']), float(v_bboxes[j]['FrontBottomLeft']['y']), float(v_bboxes[j]['FrontBottomLeft']['z']), 1.]
        points3d_world[j*8+3,:] = [float(v_bboxes[j]['FrontBottomRight']['x']), float(v_bboxes[j]['FrontBottomRight']['y']), float(v_bboxes[j]['FrontBottomRight']['z']), 1.]
        points3d_world[j*8+4,:] = [float(v_bboxes[j]['BackTopLeft']['x']), float(v_bboxes[j]['BackTopLeft']['y']), float(v_bboxes[j]['BackTopLeft']['z']), 1.]
        points3d_world[j*8+5,:] = [float(v_bboxes[j]['BackTopRight']['x']), float(v_bboxes[j]['BackTopRight']['y']), float(v_bboxes[j]['BackTopRight']['z']), 1.]
        points3d_world[j*8+6,:] = [float(v_bboxes[j]['BackBottomLeft']['x']), float(v_bboxes[j]['BackBottomLeft']['y']), float(v_bboxes[j]['BackBottomLeft']['z']), 1.]
        points3d_world[j*8+7,:] = [float(v_bboxes[j]['BackBottomRight']['x']), float(v_bboxes[j]['BackBottomRight']['y']), float(v_bboxes[j]['BackBottomRight']['z']), 1.]
        
    points3d_world = np.transpose(points3d_world)
    
    
    # Read camera matrix
    Q = np.reshape(data['intrinsic']['matrix'], [4, 4])
    M = np.reshape(data['extrinsic']['matrix'], [4, 4])
    
    # World 3d points to camrea 3d points
    points3d_camera = np.matmul(inv(M), points3d_world)
    
    # -------------------------------------------------------------------------
    
    # Finding the index of instances having positive z 
    
    nInstances = int(np.shape(points3d_camera)[1]/8)
    points3d_camera_reshape = np.reshape(points3d_camera, [4, nInstances, 8])
    
    # -------------------------------------------------------------------------
    
    # Removing exploding boxes by cutting at z = 10

#    new_z = 10
#    indx = []
#    for view in range(8):
#        indx.append(points3d_camera_reshape[2, view,:] < new_z)
#    d_vectors = np.zeros((points3d_camera_reshape.shape[1],points3d_camera_reshape.shape[0],
#                         points3d_camera_reshape.shape[2]),dtype=np.float32)
#    for view in range(4):
#        d_vectors[view,:,:] = np.squeeze(points3d_camera_reshape[:, view,:] - points3d_camera_reshape[:, view + 4,:])
#    
#    for view in range(4):
#        d_vectors[view + 4,:,:] = -d_vectors[view,:,:]
#    
#    indx = np.asarray(indx)
#    new_points3d_camera_reshape = points3d_camera_reshape
#    for view in range(8):
#        indexes = np.where(indx[view,:])
#        point = [np.divide(np.multiply(d_vectors[view, 0, indexes],new_z - points3d_camera_reshape[2, view, indexes]), d_vectors[view, 2, indexes]) + points3d_camera_reshape[0,view,indexes],
#                 np.divide(np.multiply(d_vectors[view, 1, indexes],new_z - points3d_camera_reshape[2, view, indexes]), d_vectors[view, 2,indexes]) + points3d_camera_reshape[1, view, indexes]];
#        new_points3d_camera_reshape[2, view, indexes] = new_z;
#        point = np.asarray(point)
#        new_points3d_camera_reshape[0, view, indexes] = point[0];
#        new_points3d_camera_reshape[1, view, indexes] = point[1];
#    
#    new_points3d_camera = np.reshape(new_points3d_camera_reshape, points3d_camera.shape)

    # -------------------------------------------------------------------------
    
#    points2d = np.matmul(inv(Q), new_points3d_camera)
            
    points2d = np.matmul(inv(Q), points3d_camera)
    
    points2d = np.divide(points2d, points2d[3,:])
    
    points2d[1,:] = img_size[0] - points2d[1,:] 
    
    
    nInstances = int(points2d.shape[1]/8)
    points2d_reshape = np.reshape(points2d, [4, nInstances, 8])
    
    seq_nos = instance_label_arr[:, 2].astype(int)
    class_ids = instance_label_arr[:, 1].astype(int)
    
    count = 0
    numobj = 0
    
    label_arr = []
    
    for j in seq_nos:
        
            
        # ---------------------------------------------------------------------
        
        # 2D Object Coordinates
        
        x_p_left = np.amin(points2d_reshape[0, j, :])
        
        if x_p_left < 0:
            x_p_left = 0
        if x_p_left > width -1:
            x_p_left = width - 1
            
        y_p_left = np.amin(points2d_reshape[1, j, :])
        
        if y_p_left < 0:
            y_p_left = 0 
        if y_p_left > height -1:
            y_p_left = height - 1
            
        x_p_right = np.amax(points2d_reshape[0, j, :])
        
        if x_p_right < 0:
            x_p_right = 0
        if x_p_right > width -1:
            x_p_right = width -1
            
        y_p_right = np.amax(points2d_reshape[1, j, :])
        
        if y_p_right < 0:
            y_p_right = 0
        if y_p_right > height -1:
            y_p_right = height - 1


        p_left = [x_p_left, y_p_left]
        p_right = [x_p_right, y_p_right]

        w = p_right[0] - p_left[0]
        h = p_right[1] - p_left[1]
        
        p_left[0] = round(p_left[0], 2)
        p_left[1] = round(p_left[1], 2)
        p_right[0] = round(p_right[0], 2)
        p_right[1] = round(p_right[1], 2)

        
        # Visualize
        p_left_x_int = int(p_left[0])
        p_left_y_int = int(p_left[1])
        p_right_x_int = int(p_right[0])
        p_right_y_int = int(p_right[1])
        
        #area = h * w
        #ratio =  area / (1242 * 375)
        
        ratio =  w / 1242 
        
        if ratio > 0.75:
            x_p_left = 0 + 0.30*points2d_reshape[0, j, 3]
            x_p_right = 0.87*width + 0.13*points2d_reshape[0, j, 2]
            
            y_p_left = 0 + 0.25*points2d_reshape[1, j, 0]
            y_p_right = height - 1
            
            p_left = [x_p_left, y_p_left]
            p_right = [x_p_right, y_p_right]
    
            w = p_right[0] - p_left[0]
            h = p_right[1] - p_left[1]
            
            p_left[0] = round(p_left[0], 2)
            p_left[1] = round(p_left[1], 2)
            p_right[0] = round(p_right[0], 2)
            p_right[1] = round(p_right[1], 2)
    
            
            # Visualize
            p_left_x_int = int(p_left[0])
            p_left_y_int = int(p_left[1])
            p_right_x_int = int(p_right[0])
            p_right_y_int = int(p_right[1])
        
        # ---------------------------------------------------------------------
        
        # rotation_y
        x0 = points3d_camera_reshape[0, j, 0]
        z0 = points3d_camera_reshape[2, j, 0]
        
        x4 = points3d_camera_reshape[0, j, 4]
        z4 = points3d_camera_reshape[2, j, 4]
        
        slope = (z4 - z0)/(x4 - x0)
        ry = np.arctan(slope)
        
        ry = round(ry, 2)
        
        # ---------------------------------------------------------------------
        
        # 3D Object Coordinates
        
#        x3d = points3d_camera_reshape[0, j, 4]
#        y3d = points3d_camera_reshape[1, j, 4]
#        z3d = points3d_camera_reshape[2, j, 4]

        
        x3d = 0.5*(np.amin(points3d_camera_reshape[0, j, :]) + np.amax(points3d_camera_reshape[0, j, :]))
        y3d = -np.amax(points3d_camera_reshape[1, j, :]) 
        z3d = 0.5*(np.amin(points3d_camera_reshape[2, j, :]) + np.amax(points3d_camera_reshape[2, j, :]))
        


        w3d = abs(np.amax(points3d_camera_reshape[0, j, :]) - np.amin(points3d_camera_reshape[0, j, :]))
        h3d = abs(np.amax(points3d_camera_reshape[1, j, :]) - np.amin(points3d_camera_reshape[1, j, :]))
        l3d = abs(np.amax(points3d_camera_reshape[2, j, :]) - np.amin(points3d_camera_reshape[2, j, :]))

        x3d = round(x3d, 2)
        y3d = round(y3d, 2)
        z3d = round(z3d, 2)
        
        w3d = round(w3d, 2)
        h3d = round(h3d, 2)
        l3d = round(l3d, 2)

   
        # ---------------------------------------------------------------------
        
        if w > 0:
            
            numobj = numobj +1
            
            class_id = class_ids[count]
         
            if class_id == 12:
                # Pedestrian --> Blue
                label_arr = np.append(label_arr, ['Pedestrian', 0.00, 0, ry, p_left[0], p_left[1], p_right[0], p_right[1], h3d, w3d, l3d, x3d, y3d, z3d, ry])
                
                cv.rectangle(dummy_img,(p_left_x_int,p_left_y_int),(p_right_x_int,p_right_y_int),(255,0,0),1)
            
            elif class_id == 14:
                # Car --> Green
                label_arr = np.append(label_arr, ['Car', 0.00, 0, ry, p_left[0], p_left[1], p_right[0], p_right[1], h3d, w3d, l3d, x3d, y3d, z3d, ry])
                
                cv.rectangle(dummy_img,(p_left_x_int,p_left_y_int),(p_right_x_int,p_right_y_int),(0,255,0),1)
            
            elif class_id == 19:
                # Cyclist --> Red
                label_arr = np.append(label_arr, ['Cyclist', 0.00, 0, ry, p_left[0], p_left[1], p_right[0], p_right[1], h3d, w3d, l3d, x3d, y3d, z3d, ry])
                cv.rectangle(dummy_img,(p_left_x_int,p_left_y_int),(p_right_x_int,p_right_y_int),(0,0,255),1)
    
        # ---------------------------------------------------------------------
                
        count = count + 1
        
    # -------------------------------------------------------------------------
    
    num_objects = int(len(label_arr)/15)
    label_arr = np.reshape(label_arr, [num_objects, 15])
    
    
    # Visualise 2D bounding boxes
    # comment out if necessary
    #cv.imshow('Anotated Image',dummy_img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
    return label_arr, dummy_img
    
    
# -----------------------------------------------------------------------------


# Main 
    
import os
import glob

# Read Operations --> change according to need

data_dir = "50_model_illumination"

f_rgb = glob.glob(os.path.join(data_dir, '**/RGB/*.png'), recursive=True)
f_sem = glob.glob(os.path.join(data_dir, '**/SemSeg/*.png'), recursive=True)
f_info = glob.glob(os.path.join(data_dir, '**/Information/*.json'), recursive=True)
#f_pc = glob.glob(os.path.join(data_dir, '**/LidarImages/*.json'), recursive=True)



f_rgb.sort()
f_sem.sort()
f_info.sort()
#f_pc.sort()

# Write Operations --> change according to need
  
IMG_ROOT = 'SYNTHIA_KITTI_demo/object/training/image_2/'
PC_ROOT = 'SYNTHIA_KITTI_demo/object/training/velodyne/'
CALIB_ROOT = 'SYNTHIA_KITTI_demo/object/training/calib/'
LABEL_ROOT = 'SYNTHIA_KITTI_demo/object/training/label_2/'

# Comment following if not needed
ANOT_IMG_ROOT = 'SYNTHIA_KITTI_demo/object/training/anot_image_2/'


for frame in range(38, len(f_rgb)):
    
    print("Reading Frame no:", frame, "\n")
    
    info_path = f_info[frame]
    image_path = f_rgb[frame]
    sem_path = f_sem[frame]
    #old_pc_path = f_pc[frame]

    info_path_list = info_path.split('/')
    info_no_list = info_path_list[-1].split('.')
    info_no = int(info_no_list[0]) 
    pc_no = info_no - 30
    
    pc_path_root = info_path_list[0:-2]
    pc_path_root = '/'.join(pc_path_root)
    
    pc_path = pc_path_root + '/LidarImages/' + '%06d.json' % pc_no
    # -------------------------------------------------------------------------
    
    # Load lidar points 
    points_homg_camera, instance_ids = load_lidar_points(pc_path, info_path)
    
    # Write point cloud
    
    out_pc_path = PC_ROOT + '%06d.bin' % frame
    points_homg_camera.astype('float32').tofile(out_pc_path)
    
    # -------------------------------------------------------------------------
    
    # Load camera parameters
    P, R, velo_to_cam = load_calib(info_path)
    
    # Write camera parameters

    out_calib_path = CALIB_ROOT + '%06d.txt' % frame
    
    P0_str = 'P0:' 
    P1_str = 'P1:'
    P2_str = 'P2:'
    P3_str = 'P3:'
    R_str = 'R0_rect:'
    velo_to_cam_str = 'Tr_velo_to_cam:'
    imu_to_velo_str = 'Tr_imu_to_velo:'
    
    
    for i in range(12):
        P0_str = P0_str + ' ' + str(P[0, i])
        P1_str = P1_str + ' ' + str(P[0, i])
        P2_str = P2_str + ' ' + str(P[0, i])
        P3_str = P3_str + ' ' + str(P[0, i])
        velo_to_cam_str = velo_to_cam_str + ' ' + str(velo_to_cam[0, i])
        imu_to_velo_str = imu_to_velo_str + ' ' + str(velo_to_cam[0, i])
    
    for i in range(9):
        R_str = R_str + ' ' + str(R[0, i])
    
    
    calibfile = open(out_calib_path,'w+')
    
    calibfile.write(P0_str)
    calibfile.write('\n')
    calibfile.write(P1_str)
    calibfile.write('\n')
    calibfile.write(P2_str)
    calibfile.write('\n')
    calibfile.write(P3_str)
    calibfile.write('\n')
    calibfile.write(R_str)
    calibfile.write('\n')
    calibfile.write(velo_to_cam_str)
    calibfile.write('\n')
    calibfile.write(imu_to_velo_str)
    calibfile.write('\n')
    
    calibfile.close()
    
    # -------------------------------------------------------------------------
    
    # Generate instance-label array
    instance_label_arr = instance_label_arr_generator(info_path, sem_path)
    
    # -------------------------------------------------------------------------
    
    # Generate label array
    label_arr, anot_img = labelarr_generator(info_path, image_path, instance_label_arr)
    
    # Write label array
    
    out_label_path = LABEL_ROOT + '%06d.txt' % frame
    
    labelfile = open(out_label_path,'w+')
    
    m,n = np.shape(label_arr)
    
    for i in range(m):
        for j in range(n):
            labelfile.write(str(label_arr[i, j]))
            labelfile.write(' ')
        labelfile.write('\n')
            
    labelfile.close()
    
    # Visualizing annotations --> comment if not needed
    anot_img_path = ANOT_IMG_ROOT + '%06d.png' % frame
    cv.imwrite(anot_img_path, anot_img)
    # -------------------------------------------------------------------------
    
    # Write (copy) rgb image
    
    out_img_path = IMG_ROOT + '%06d.png' % frame
    shutil.copy2(image_path, out_img_path)
    
    # -------------------------------------------------------------------------
