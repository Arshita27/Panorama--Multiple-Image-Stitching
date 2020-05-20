import argparse
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from config import ConfigNode
from image_stitching import ImageStitching


def main(cfg):

    panaroma = ImageStitching(cfg)
    all_raw_image_list = panaroma.read_image()

    count = 1
    while len(all_raw_image_list) > 1:

        raw_image_list = all_raw_image_list[0:2]
        all_raw_image_list.pop(0)
        all_raw_image_list.pop(0)

        gray_image_list = panaroma.convert_2_gray(raw_image_list)

        kps_list, descs_list = panaroma.get_feature_desc(gray_image_list)

        panaroma.draw_keypoints(raw_image_list, gray_image_list, kps_list, descs_list, )

        matches = panaroma.get_best_matches(gray_image_list[0:2], kps_list[0:2], descs_list[0:2], )

        H = panaroma.get_homography_matrix(matches, kps_list[0:2])

        dst = panaroma.get_warped_image(raw_image_list[0:2], H)

        dst[0:raw_image_list[0].shape[0], 0:raw_image_list[0].shape[1]] = raw_image_list[0]

        dst = panaroma.contour_fitting(dst)

        dst = dst.astype('uint8')
        cv2.imwrite(os.path.join(cfg.DATASET.OUTPUT_DIR, str(count) + 'output.jpg'),dst)

        all_raw_image_list = [dst] + all_raw_image_list

        count+=1

    cv2.imwrite(os.path.join(cfg.DATASET.OUTPUT_DIR, 'final_output.jpg'), dst)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument( "--config_file", required=True, help = "path to config file")

    args = parser.parse_args()

    with open(args.config_file, "r") as ymlfile:
        node = yaml.load(ymlfile)

    cfg = ConfigNode(node)
    main(cfg)
