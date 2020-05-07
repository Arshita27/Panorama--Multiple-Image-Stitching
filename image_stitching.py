import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageStitching():
    def __init__(self, cfg ):
        self.cfg = cfg
        self.root_path = cfg.DATASET.INPUT_DIR
        self.result_dir = cfg.DATASET.OUTPUT_DIR

    def read_image(self, ):
        '''
        Reads raw image.
        '''

        return [cv2.imread(os.path.join(self.root_path, file_path))
                for file_path in self.cfg.DATASET.INPUT_IMG_LIST]

    def convert_2_gray(self, img_list:List):
        '''
        Converts image into gray scale.
        '''

        return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]

    def get_feature_desc(self, img_list:List):
        '''
        Feature descriptor
        '''

        kps_list=[]
        descs_list=[]
        for img in img_list:
            if self.cfg.FEATURES.FEATURE_DESCRIPTORS == "sift":
                sift = cv2.xfeatures2d.SIFT_create()
                (kps, descs) = sift.detectAndCompute(img, None)
                kps_list.append(kps)
                descs_list.append(descs)
            elif self.cfg.FEATURES.FEATURE_DESCRIPTORS == "surf":
                pass
                # NOTE: TODO
            else:
                pass
                # NOTE: Raise Error (Feature Descriptor not found)
        return kps_list, descs_list

    def draw_keypoints(self, raw_img_list, gray_img_list, kps_list, descs_list):
        '''
        '''

        for i, raw_img in enumerate(raw_img_list):
            raw_img_copy = raw_img.copy()
            img_keypoints = cv2.drawKeypoints(
                                    gray_img_list[i],
                                    kps_list[i],
                                    raw_img_copy,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(os.path.join(self.result_dir, "img_"+str(i)+"_keyfeatures.png"),
                        img_keypoints)
        print(f'Saving images with keypoints.')

    def get_best_matches(self, img, kps, descs, T:float=0.5):
        '''
        Feature Matching
        '''

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descs[0], descs[1], k=2)
        good = []
        for m,n in matches:
            if m.distance < T*n.distance:
                good.append([m])

        res_img = cv2.drawMatchesKnn(img[0], kps[0], img[1], kps[1], good, None, flags=2)
        cv2.imwrite(os.path.join(self.result_dir, 'matched_points.jpg'), res_img)

        return np.asarray(good)

    def get_homography_matrix(self, matches, kps):
        '''
        '''

        if len(matches[:,0]) >= 4:
            dst = np.float32([ kps[0][m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            src = np.float32([ kps[1][m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        return H

    def get_warped_image(self, img, H):
        '''
        '''

        dst = cv2.warpPerspective(img[1], H,
                                 (img[1].shape[1] + img[0].shape[1], img[0].shape[0]),
                                 )
        cv2.imwrite(os.path.join(self.result_dir, 'warped.jpg'), dst)

        return dst


    def detect_corners_from_contour(self, contour):
        """
        Detecting corner points form contours using cv2.approxPolyDP()
        Args:
            contour: list
        Returns:
            approx_corners: np.array
        """

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_corners = cv2.approxPolyDP(contour, epsilon, True)
        approx_corners = sorted(np.concatenate(approx_corners).tolist())
        approx_corners = np.array([approx_corners[i] for i in [0, 2, 1, 3]], np.float32)

        return approx_corners


    def contour_fitting(self, img):

        w = img.shape[1]
        h = img.shape[0]
        img_shape = np.array([[0,0], [w,0], [0,h], [w, h]], np.float32)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,3)

        ret,thresh = cv2.threshold(gray,1,255,0)
        _, contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        img_copy = img.copy()
        img_copy = cv2.drawContours(img_copy, contours, -1, (0,255,0), 3)

        approx_corners = self.detect_corners_from_contour(contours[0])

        M  = cv2.getPerspectiveTransform(approx_corners, img_shape)
        new_image = cv2.warpPerspective(img, M, (w, h))

        return new_image
