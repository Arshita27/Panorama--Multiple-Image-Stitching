"""
NOTE: This is a slight modification of the code from:
https://github.com/cynricfu/multi-band-blending
"""

import numpy as np
import cv2
import sys
import argparse


def preprocess(img1, img2, overlap_w):

    w1 = img1.shape[1]
    w2 = img2.shape[1]

    shape = np.array(img1.shape)
    shape[1] = w1 + w2 - overlap_w

    subA = np.zeros(shape)
    subA[:, :w1] = img1
    subB = np.zeros(shape)
    subB[:, w1 - overlap_w:] = img2
    mask = np.zeros(shape)

    if overlap_w %2 != 0:
        overlap_w_half = int(overlap_w/2)-1
    else:
        overlap_w_half = int(overlap_w/2)
    mask[:, :w1 - overlap_w_half] = 1

    return subA, subB, mask


def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP


def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):

        next_img = cv2.pyrDown(img)

        pyrUp_next_img = cv2.pyrUp(next_img, img.shape[1::-1])
        pyrUp_next_img = cv2.resize(pyrUp_next_img, img.shape[1::-1])
        LP.append(img - pyrUp_next_img)

        img = next_img
    LP.append(img)
    return LP


def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended


def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, lev_img.shape[1::-1])
        img = cv2.resize(img, lev_img.shape[1::-1])
        img += lev_img
    return img


def multi_band_blending(img1, img2, overlap_w, leveln=None, ):
    if overlap_w < 0:
        print ("error: overlap_w should be a positive integer")
        sys.exit()

    subA, subB, mask = preprocess(img1, img2, overlap_w,)

    max_leveln = int(np.floor(np.log2(min(img1.shape[0], img1.shape[1],
                                          img2.shape[0], img2.shape[1]))))
    if leveln is None:
        leveln = max_leveln
    if leveln < 1 or leveln > max_leveln:
        print ("warning: inappropriate number of leveln")
        leveln = max_leveln

    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(subA, leveln)
    LPB = LaplacianPyramid(subB, leveln)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    result[result > 255] = 255
    result[result < 0] = 0

    return result
