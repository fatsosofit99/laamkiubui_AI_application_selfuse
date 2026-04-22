import cv2
import numpy as np

def load_image_hsv(image_path: str) -> np.ndarray:
    #TODO
    img=cv2.imread(image_path)
    img_hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    return img_hsv
def is_medical_report(im_hsv: np.ndarray, threshold_s: float, threshold_v: float) -> bool:
    #TODO
    mean_s = np.mean(im_hsv[:,:,1])
    mean_v = np.mean(im_hsv[:,:,2])
    return mean_s<threshold_s and mean_v>threshold_v
if __name__ == '__main__':
    im = load_image_hsv('/home/project/example.png')
    result = is_medical_report(im, 150, 50)
    print(result)