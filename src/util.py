import cv2 as cv

def read_img(filename):
    img = cv.imread(filename)
    if img is None:
        raise FileNotFoundError(f"File {filename} not found")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
    return img
