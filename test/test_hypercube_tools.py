import unittest
import os

from src.utils.hypercube_tools import HypercubeTools

DATASETS_PATH = "src/datasets/train"
PATH_SEGMENTED = DATASETS_PATH + "/GT_color/"
RGB_PATH = DATASETS_PATH + "/rgb/"
NIR_PATH = DATASETS_PATH + "/nir_color/"
EVI_PATH = DATASETS_PATH + "/evi_color/"
GT_PATH = DATASETS_PATH +  "/gt_files/"


class TestHypercubeTools(unittest.TestCase):

    def test_create_gt_for_images(self):
        path = "src/datasets/train/GT_color/b1-99445_mask.png"
        segmented_image = HypercubeTools.read_image(path)
        gt_img = HypercubeTools.create_gt_for_images(segmented_image)
        self.assertEqual(len(gt_img.shape), 2)

    def test_create_hypercubes_list(self):
        list_length=50
        hypercube_tools = HypercubeTools()
        hypercubes_list = hypercube_tools.create_hypercubes_list(PATH_SEGMENTED, RGB_PATH, NIR_PATH, EVI_PATH, GT_PATH, list_length=list_length)
        self.assertEqual(len(hypercubes_list), len(hypercubes_list))
    

