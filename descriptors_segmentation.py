from src.utils.hypercube_tools import HypercubeTools
import cv2

if __name__ == "__main_":
    hypercube_tools = HypercubeTools()
    img = hypercube_tools.read_image("src/datasets/results")
    cv2.imshow(img*(255/5))