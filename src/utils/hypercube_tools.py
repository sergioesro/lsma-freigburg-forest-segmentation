import numpy as np
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import pickle as pk

from src.utils.image_descriptors import RGBHistogram


COLOR_DICT = {
            "vegetation": [51, 102, 102],
            "grass": [0, 255, 0],
            "path": [170, 170, 170],
            "sky": [255, 120,   0],
            "object": [0, 60, 0]
            }   

GT_DICT = {
            0: [51, 102, 102],
            1: [0, 255, 0],
            2: [170, 170, 170],
            3: [255, 120,   0],
            4: [0, 60, 0]
        }

class HypercubeTools:

    def create_hypercubes_list(self, path_segmented, rgb_path, nir_path, evi_path, gt_path, list_length=None):
        # So they are taken in the same order
        rgb_list = sorted(os.listdir(rgb_path))
        nir_list = sorted(os.listdir(nir_path))
        evi_list = sorted(os.listdir(evi_path))
        gt_list = sorted(os.listdir(gt_path))

        hypercubes_list = []

        files = sorted(os.listdir(path_segmented))
        for idx, elem in enumerate(files):
            print("Filename files: ", elem)
            rgb_img = cv2.imread(f"{rgb_path}{rgb_list[idx]}")
            nir_img = cv2.imread(f"{nir_path}{nir_list[idx]}")
            evi_img = cv2.imread(f"{evi_path}{evi_list[idx]}")
            if gt_path: # If ground truth is already created we don't waste time in creating them (computationally demanding)
                gt_img = np.load(f"{gt_path}{gt_list[idx]}")
            else:
                img_segmented = cv2.imread(f"{path_segmented}{elem}")
                gt_img = self.create_gt_for_images(img_segmented,name=nir_list[idx][:-4], save=False)
            hypercube = self.create_final_hypercube(rgb_img, nir_img, evi_img, gt_img)
            hypercubes_list.append(hypercube)
            if list_length is not None and idx == list_length:
                print(f"Succesfully created list of {list_length} hypercubes")
                break
        return hypercubes_list
    
    def create_final_hypercube(self, rgb_img, nir_img, evi_img, gt_img):
        hypercube = self.append_bands_hypercube(rgb_img,nir_img)
        hypercube = self.append_bands_hypercube(hypercube,evi_img)
        hypercube = self.append_bands_hypercube(hypercube,gt_img)
        return hypercube
    
    def create_small_hypercube(self, rgb_img, evi_img, gt_img):
        hypercube = self.append_bands_hypercube(rgb_img,evi_img)
        hypercube = self.append_bands_hypercube(hypercube,gt_img)
        return hypercube

    
    def get_pixels_from_gt(self, hypercube, gt_value, num_bands=3, save=False, name=None):
        '''
        At first focused on hypercubes and takes pixel values corresponding to the specified gt value on the image
        '''
        matrix = np.float16(self.matricization_mode_3(hypercube, debug=True))
        # pixels_gt = np.zeros((num_bands,1))
        # for i in range(matrix.shape[1]):
        #     if int(matrix[-1,i]) == gt_value and isinstance(gt_value,int):
        #         if not np.any(pixels_gt):
        #             pixels_gt = matrix[0:num_bands,i]
        #         else:
        #             pixels_gt = np.vstack((pixels_gt, matrix[0:num_bands,i]))
        for i in range(10):
            idx_low = int(matrix.shape[1]/10)*i + 1
            idx_up = int(matrix.shape[1]/10)*(i+1) + 1
            if i == 0:
                pixels_gt = np.squeeze(matrix[:-1, np.where(matrix[-1,idx_low:idx_up]==gt_value)])
            else:
                pixels_gt = np.hstack((pixels_gt, np.squeeze(matrix[:-1, np.where(matrix[-1,idx_low:idx_up]==gt_value)])))
        if save and name is not None:
            np.save(f'src/datasets/gt_pixels/pixel_{gt_value}_{name}_img.npy', pixels_gt)
        return pixels_gt
    

    def segmentate_using_descriptors(self, hypercube, descriptor, features_list, mode="rgb",
                                      save = False, name=None):
        '''
        Create segmentation image using the specified descriptor
        '''
        segmentation = np.zeros((hypercube.shape[0], hypercube.shape[1]))
        for i in range(hypercube.shape[0]):
            for j in range (hypercube.shape[1]):
                differences = []
                pixel_val = hypercube[i,j,0:3]
                pixel = Image.fromarray(np.uint8(pixel_val * 255))
                pixel.save('src/datasets/gt_pixels/pixel_aux.png')
                pix_img = 'src/datasets/gt_pixels/pixel_aux.png'
                img_pix = cv2.imread(pix_img)
                features_pixel = descriptor.describe(img_pix)
                for feature in features_list:
                    desc_difference = np.abs(feature - features_pixel)
                    differences.append(desc_difference)
                sums = [np.sum(diff) for diff in differences]
                class_index = np.argmin(sums)
                segmentation[i, j] = class_index
        if save and name is not None:
            segmentation_aux = Image.fromarray(np.uint8(segmentation * (255/5)))
            segmentation_aux.save(f'src/datasets/results/segmentation_{name}.png')
        return segmentation
    

    @staticmethod
    def calculate_distances_to_index(hypercube, features_list, save=False, name=None):
        out_hypercube = np.zeros((hypercube.shape[0], hypercube.shape[1]))
        for i in range(hypercube.shape[0]):
            for j in range(hypercube.shape[1]):
                differences = []
                pixel_val = hypercube[i,j,:]
                for el in features_list:
                    distances = np.linalg.norm(el.T - pixel_val.T, axis=1)
                    differences.append(np.mean(distances))
                out_hypercube[i,j] = np.argmin(differences)
        if save and name is not None:
            segmentation_aux = Image.fromarray(np.uint8(out_hypercube * (255/5)))
            segmentation_aux.save(f'src/datasets/results/segmentation_{name}.png')
        return out_hypercube


    @staticmethod
    def create_gt_for_images(segmented_image, name=None, save=False, debug=False):
        gt_img = np.zeros((segmented_image.shape[0],segmented_image.shape[1]))
        for i in range(len(segmented_image[:,1,1])):
            for j in range(len(segmented_image[1,:,1])):
                idx = 0
                for el in list(GT_DICT.values()):
                    if all(segmented_image[i,j,:] == el):
                        gt_img[i,j] = idx
                    else:
                        idx += 1
        if save and name is not None:
            # TODO: Change this to local but it is already created
            np.save(f"/content/gdrive/MyDrive/LSMA_Final_Project/datasets/freiburg_forest_annotated/train/gt_files/{name}",
                    gt_img)
        return gt_img
    
    @staticmethod
    def append_bands_hypercube(img, bands_to_append, debug=False):
        hypercube = np.dstack((img,bands_to_append))
        if debug:
            print(f"Shape of hypercube is {hypercube.shape}")
        return hypercube

    # To simplify EDA
    @staticmethod
    def matricization_mode_3(hypercube, debug=False):
        matrix = hypercube.transpose(2,0,1).reshape(hypercube.shape[2],-1)
        if debug:
            print(f'Shape of matrix is {matrix.shape}')
        return matrix
    

    @staticmethod
    def pca(matrix_pca, n_components= 2, pca_name=None):
        std_slc = StandardScaler()
        X = matrix_pca[:-1, :]
        std_slc.fit(X)
        X_scaled = std_slc.transform(X)
        n_components = matrix_pca.shape[0] - 1
        pca = decomposition.PCA(n_components=n_components)
        X_std_pca = pca.fit(X_scaled)
        return X_std_pca
    
    @staticmethod
    def scale_bands(hypercube):
        hypercube_bands_scaled = np.zeros((hypercube.shape[0], hypercube.shape[1], hypercube.shape[2]-1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        for i in range(hypercube.shape[-1]-1):
            band = hypercube[:, :, i]
            band_scaled = scaler.fit_transform(band.reshape(-1, 1)).reshape(band.shape)
            hypercube_bands_scaled[:, :, i] = band_scaled
        return hypercube_bands_scaled

    @staticmethod
    def pca_return_hypercube(hypercube_bands, n_comp=3, use_pca=False, pca_name=None, save_pca=False, save_name=None):
        bands_2d = hypercube_bands.reshape(-1, hypercube_bands.shape[2]) 
        if use_pca and pca_name is not None:
            pca = pk.load(open(f'src/datasets/pca_models/{pca_name}.pkl','rb'))
        else:
            pca = PCA(n_components=n_comp).fit(bands_2d)
        if save_pca and save_name is not None:
            pk.dump(pca, open(f'src/datasets/pca_models/{save_name}.pkl', 'wb'))
        bands_pca = pca.transform(bands_2d)
        bands_pca = bands_pca.reshape(hypercube_bands.shape[0], hypercube_bands.shape[1], -1)
        explained_variance = pca.explained_variance_ratio_
        print("The explained variance of the first", n_comp ,"components is: ", sum(explained_variance[:n_comp]))
        return bands_pca

    @staticmethod
    def translate_from_gt_to_rgb(gt_image, save=False, name=None):
        '''
        Translate from our grayscale image to our dictionary
        '''
        rgb_gt = np.zeros((gt_image.shape[0], gt_image.shape[1], 3))
        if np.max(gt_image) >= 204:
            gt_image = gt_image*(5/255)
        for i in range(gt_image.shape[0]): # len(gt_image[:,1,1])
            for j in range(gt_image.shape[1]): # len(gt_image[1,:,1])
                if len(gt_image.shape) == 3:
                    gt_px = gt_image[i,j,:]
                    rgb_gt[i,j,:] = np.array(GT_DICT[int(gt_px[0])])
                else:
                    gt_px = gt_image[i,j]
                    rgb_gt[i,j,:] = np.array(GT_DICT[int(gt_px)])
        if save and name is not None:
            # rgb_aux = Image.fromarray(np.uint8(rgb_gt))
            cv2.imwrite(f'src/datasets/results/img_gt_{name}.png', rgb_gt)
        return rgb_gt
    

    @staticmethod
    def read_image(img_path):
        '''
        Simply reads an image
        '''
        img = cv2.imread(img_path)
        return img
    