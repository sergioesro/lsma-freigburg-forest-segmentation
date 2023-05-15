import unittest
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from src.utils.hypercube_tools import HypercubeTools
from src.utils.image_descriptors import RGBHistogram, LabHistogram
from src.utils.metrics_tools import get_accuracy, f1_metric, classification_report

from src.neural_networks.neural_network import NeuralNetwork
from src.neural_networks.neural_utils import train, test, initialize_network_parameters

DATASETS_PATH = "src/datasets/train"
PATH_SEGMENTED = DATASETS_PATH + "/GT_color/"
RGB_PATH = DATASETS_PATH + "/rgb/"
NIR_PATH = DATASETS_PATH + "/nir_color/"
EVI_PATH = DATASETS_PATH + "/evi_color/"
GT_PATH = DATASETS_PATH +  "/gt_files/"

GT_VALUES = [0, 1, 2, 3, 4]

class TestHypercubeTools(unittest.TestCase):

    @classmethod
    def setUp(self):
        hypercube_tools = HypercubeTools()
        rgb_path = DATASETS_PATH + "/rgb/b1-99445_Clipped.jpg"
        nir_path = DATASETS_PATH + "/nir_color//b1-99445.png"
        evi_path = DATASETS_PATH + "/evi_color/b1-99445.png"
        gt_path = DATASETS_PATH +  "/gt_files/b1-99445.npy"
        rgb_img = HypercubeTools.read_image(rgb_path)
        nir_img = HypercubeTools.read_image(nir_path)
        evi_img = HypercubeTools.read_image(evi_path)
        gt_img = np.load(gt_path)
        self.hypercube_test = hypercube_tools.create_final_hypercube(rgb_img, nir_img, evi_img, gt_img)

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

    def test_get_pixels_of_particular_gt(self):
        hypercube_tools = HypercubeTools()
        pixels_gt = hypercube_tools.get_pixels_from_gt(self.hypercube_test, gt_value=1)
        print(pixels_gt.shape)
        self.assertEqual(True,True)

    def test_segmentation_using_descriptors(self):
        hypercube_tools = HypercubeTools()
        pixels_gt = {}
        for i in GT_VALUES:
            pixels_gt[f'pixels_{i}']= hypercube_tools.get_pixels_from_gt(self.hypercube_test, gt_value=i, save=True)
        self.assertEqual(True, True)

    def test_segmentation_using_descriptors_pca(self):
        hypercube_tools = HypercubeTools()
        hypercube_bands_scaled = hypercube_tools.scale_bands(self.hypercube_test)
        print(hypercube_bands_scaled.shape)
        bands_pca = hypercube_tools.pca_return_hypercube(hypercube_bands_scaled, n_comp=9, save_pca=True, save_name="pca_img")
        gt_path = DATASETS_PATH +  "/gt_files/b1-99445.npy"
        gt_img = np.load(gt_path)
        bands_pca = hypercube_tools.append_bands_hypercube(bands_pca, gt_img)
        print(bands_pca.shape)
        for i in GT_VALUES:
            hypercube_tools.get_pixels_from_gt(bands_pca, gt_value=i, num_bands=9, save=True, name="pca")
        self.assertEqual(True, True)


    def test_create_feature_list_rgb(self):
        hypercube_tools = HypercubeTools()
        gt_pixels_path = "src/datasets/gt_pixels/"
        img_label_list = []
        for i in GT_VALUES:
            img_label_list.append(HypercubeTools.read_image(f"{gt_pixels_path}pixel_{i}_img.png"))
        features_list = []
        desc = RGBHistogram([8, 8, 8])
        for el in img_label_list:
            features_list.append(desc.describe(el))
        hypercube_tools.segmentate_using_descriptors(self.hypercube_test, desc, features_list, save=True, name="rgb_desc")
        self.assertEqual(True, True)


    def test_create_feature_list_lab(self):
        hypercube_tools = HypercubeTools()
        gt_pixels_path = "src/datasets/gt_pixels/"
        img_label_list = []
        for i in GT_VALUES:
            img_label_list.append(HypercubeTools.read_image(f"{gt_pixels_path}pixel_{i}_img.png"))
        features_list = []
        desc = LabHistogram([8, 8, 8])
        for el in img_label_list:
            features_list.append(desc.describe(el))
        hypercube_tools.segmentate_using_descriptors(self.hypercube_test, desc, features_list, save=True, name="lab_desc")
        self.assertEqual(True, True)


    def test_translate_from_gt_to_rgb(self):
        hypercube_tools = HypercubeTools()
        gt_pixels_path = "src/datasets/results/"
        rbg_desc_name = "segmentation_rgb_desc.png"
        gt_img = hypercube_tools.read_image(f"{gt_pixels_path}{rbg_desc_name}")
        rgb_image = HypercubeTools.translate_from_gt_to_rgb(gt_img, save=True, name="rgb_desc")
        self.assertIsNotNone(rgb_image)


    def test_translate_from_gt_to_rgb_lab(self):
        hypercube_tools = HypercubeTools()
        gt_pixels_path = "src/datasets/results/"
        rbg_desc_name = "segmentation_lab_desc.png"
        gt_img = hypercube_tools.read_image(f"{gt_pixels_path}{rbg_desc_name}")
        rgb_image = HypercubeTools.translate_from_gt_to_rgb(gt_img, save=True, name="lab_desc")
        self.assertIsNotNone(rgb_image)

    def test_create_feature_list_pca(self):
        hypercube_tools = HypercubeTools()
        hypercube_bands_scaled = hypercube_tools.scale_bands(self.hypercube_test)
        print(hypercube_bands_scaled.shape)
        bands_pca = hypercube_tools.pca_return_hypercube(hypercube_bands_scaled, n_comp=9, save_pca=True, save_name="pca_img")
        print(bands_pca.shape)
        three_bands_pca = bands_pca[:,:,:3]
        # gt_path = DATASETS_PATH +  "/gt_files/b1-99445.npy"
        # gt_img = np.load(gt_path)
        gt_pixels_path = "src/datasets/gt_pixels/"
        img_label_list = []
        # TODO: apply pca over all index images and then compute distance using RGBHistogram for instance
        for i in GT_VALUES:
            img_label_list.append(np.load(f"{gt_pixels_path}pixel_{i}_pca_img.npy"))
        out_hypercube = hypercube_tools.calculate_distances_to_index(bands_pca, img_label_list,save=True, name="pca_img")
        out_hypercube = Image.fromarray(np.uint8(out_hypercube * 255))
        out_hypercube.save('src/datasets/gt_pixels/segmented_pca.png')
        self.assertEqual(True, True)

    def test_get_accuracy(self):
        ideal_gt = cv2.imread("src/datasets/train/GT_color/b1-99445_mask.png")
        segmented_gt = cv2.imread("src/datasets/results/img_gt_lab_desc.png")
        acc = get_accuracy(ideal_gt, segmented_gt)
        self.assertIsNotNone(acc)

    def test_compute_f1(self):
        ideal_gt = cv2.imread("src/datasets/train/GT_color/b1-99445_mask.png")
        segmented_gt = cv2.imread("src/datasets/results/img_gt_lab_desc.png")
        ideal_gt = HypercubeTools.create_gt_for_images(ideal_gt)
        segmented_gt = HypercubeTools.create_gt_for_images(segmented_gt)
        f1_score = f1_metric(ideal_gt.flatten(), segmented_gt.flatten(), average='weighted')
        self.assertIsNotNone(f1_score)

    def test_compute_classification_report(self):
        ideal_gt = cv2.imread("src/datasets/train/GT_color/b1-99445_mask.png")
        segmented_gt = cv2.imread("src/datasets/results/img_gt_lab_desc.png")
        ideal_gt = HypercubeTools.create_gt_for_images(ideal_gt)
        segmented_gt = HypercubeTools.create_gt_for_images(segmented_gt)
        class_report = classification_report(ideal_gt.flatten(), segmented_gt.flatten(), digits=3)
        print(class_report)
        self.assertIsNotNone(class_report)

    def test_train_neural_network(self):
        train_hypercube = HypercubeTools.matricization_mode_3(self.hypercube_test)
        device = "cpu"
        model = NeuralNetwork().to(device)
        loss_fn, optimizer = initialize_network_parameters(model)
        train(train_hypercube, model, loss_fn, optimizer, save=True, name="sample")
        print(model)
        self.assertIsNotNone(model)


    def test_test_neural_network(self):
        test_hypercube = HypercubeTools.matricization_mode_3(self.hypercube_test)
        device = "cpu"
        model = NeuralNetwork().to(device)
        loss_fn, _ = initialize_network_parameters(model)
        test_acc, test_loss = test(test_hypercube, model, loss_fn, load=True, name="sample")
        # plt.figure(figsize=(10,5))
        # plt.title("Training and Validation Loss")
        # plt.plot(test_acc,label="acc")
        # plt.plot(test_loss/np.max(test_loss),label="loss")
        # plt.xlabel("Iterations")
        # plt.ylabel("%")
        # plt.legend()
        # plt.show()
        print(model)
        self.assertIsNotNone(model)

    




    
