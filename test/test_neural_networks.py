import unittest
import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils.hypercube_tools import HypercubeTools

from src.neural_networks.neural_network import NeuralNetwork, Net_KFold, CNN
from src.neural_networks.neural_utils import train, train_kfold, train_cnn, test, initialize_network_parameters

DATASETS_PATH = "src/datasets/train"
PATH_SEGMENTED = DATASETS_PATH + "/GT_color/"
RGB_PATH = DATASETS_PATH + "/rgb/"
NIR_PATH = DATASETS_PATH + "/nir_color/"
EVI_PATH = DATASETS_PATH + "/evi_color/"
GT_PATH = DATASETS_PATH +  "/gt_files/"

GT_VALUES = [0, 1, 2, 3, 4]

class TestNeuralNetworks(unittest.TestCase):

    @classmethod
    def setUp(self):
        hypercube_tools = HypercubeTools()
        # Image 1
        rgb_path = DATASETS_PATH + "/rgb/b1-99445_Clipped.jpg"
        nir_path = DATASETS_PATH + "/nir_color//b1-99445.png"
        evi_path = DATASETS_PATH + "/evi_color/b1-99445.png"
        gt_path = DATASETS_PATH +  "/gt_files/b1-99445.npy"
        rgb_img = HypercubeTools.read_image(rgb_path)
        nir_img = HypercubeTools.read_image(nir_path)
        evi_img = HypercubeTools.read_image(evi_path)
        gt_img = np.load(gt_path)

        # Image 2
        rgb_path2 = DATASETS_PATH + "/rgb/b2-05528_Clipped.jpg"
        nir_path2 = DATASETS_PATH + "/nir_color//b2-05528.png"
        evi_path2 = DATASETS_PATH + "/evi_color/b2-05528.png"
        gt_path2 = DATASETS_PATH +  "/gt_files/b2-05528.npy"
        rgb_img2 = HypercubeTools.read_image(rgb_path2)
        nir_img2 = HypercubeTools.read_image(nir_path2)
        evi_img2 = HypercubeTools.read_image(evi_path2)
        gt_img2 = np.load(gt_path2)

        self.hypercube_test = hypercube_tools.create_final_hypercube(rgb_img, nir_img, evi_img, gt_img)
        self.hypercube_test2 = hypercube_tools.create_final_hypercube(rgb_img2, nir_img2, evi_img2, gt_img2)

    def test_train_neural_network(self):
        train_hypercube = HypercubeTools.matricization_mode_3(self.hypercube_test)
        device = "cuda"
        model = NeuralNetwork().to(device)
        loss_fn, optimizer = initialize_network_parameters(model, lr=1e-4)
        train(train_hypercube, model, loss_fn, optimizer, epochs=1, normalize=True, save=False, name="model_batch_norm", plot=True)
        print(model)
        self.assertIsNotNone(model)

    def test_train_kfolds(self):
        train_hypercube = HypercubeTools.matricization_mode_3(self.hypercube_test)
        device = "cpu"
        model = Net_KFold(num_classes=5).to(device)
        loss_fn, optimizer = initialize_network_parameters(model, lr=1e-4)
        train_kfold(train_hypercube, model, loss_fn, optimizer, epochs=20, k_folds=2)
        print(model)
        self.assertIsNotNone(model)

    def test_train_CNN(self):
        train_hypercubes = [self.hypercube_test, self.hypercube_test2]
        model = CNN()
        loss_fn, optimizer = initialize_network_parameters(model, lr=1e-6)
        train_cnn([train_hypercubes], model, loss_fn, optimizer, save=True, name='model_cnn')
        print(model)
        self.assertIsNotNone(model)

    def test_test_neural_network(self):
        gt_pixels_path = "src/datasets/results/"
        hypercube = self.hypercube_test2
        test_hypercube = HypercubeTools.matricization_mode_3(hypercube)
        device = "cpu"
        model = NeuralNetwork().to(device)
        loss_fn, _ = initialize_network_parameters(model)
        _, _, y_out = test(test_hypercube, hypercube.shape, model, loss_fn, normalize=True, load=True, name="model_batch_norm_25", generate_image=True)
        rgb_y = HypercubeTools.translate_from_gt_to_rgb(y_out, save=True, name="batch_nn_test")
        print(model)
        self.assertIsNotNone(model)

    def test_full_training_epochs(self):
        test_accs = []
        test_losses = []
        hypercube = HypercubeTools.matricization_mode_3(self.hypercube_small)
        device = "cpu"
        model = NeuralNetwork().to(device)
        loss_fn, optimizer = initialize_network_parameters(model, lr=1e-4)
        train(hypercube, model, loss_fn, optimizer, epochs=3, save=False, name="sample_small")
        test_acc, test_loss = test(hypercube, model, loss_fn, load=True, name="sample_normalized")
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        plt.figure(figsize=(10,5))
        plt.title("Training and Validation Loss")
        plt.plot(test_accs,label="acc")
        plt.plot(test_losses/np.max(test_loss),label="loss")
        plt.xlabel("Iterations")
        plt.ylabel("%")
        plt.legend()
        plt.show()
        print("Finished training!")