import numpy as np
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

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
            hypercube = self.append_bands_hypercube(rgb_img,nir_img)
            hypercube = self.append_bands_hypercube(hypercube,evi_img)
            hypercubes_list.append(self.append_bands_hypercube(hypercube,gt_img))
            if list_length is not None and idx == list_length:
                print(f"Succesfully created list of {list_length} hypercubes")
                break
        return hypercubes_list

    def read_image(img_path):
        img = cv2.imread(img_path)
        return img

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
    def matricization_mode_3(hypercube, debug=False):
        matrix = hypercube.transpose(2,0,1).reshape(hypercube.shape[2],-1)
        if debug:
            print(f'Shape of matrix is {matrix.shape}')
        return matrix
    
    def pca(matrix_pca, n_components= 2):
        std_slc = StandardScaler()
        X = matrix_pca[:-1, :]
        std_slc.fit(X)
        X_scaled = std_slc.transform(X)
        n_components = matrix_pca.shape[0] - 1
        pca = decomposition.PCA(n_components=n_components)
        X_std_pca = pca.fit(X_scaled)
        return X_std_pca