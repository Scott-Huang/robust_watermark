import numpy as np
import cv2 as cv
from functools import reduce

class Distribution:
    def sample(self):
        return None
    pass

class Filter:
    def __init__(self, pts_dist, params_dist):
        self.pts_dist = pts_dist
        self.params_dist = params_dist
        self.sub_filters = []

    def apply(self, image):
        return image + reduce(lambda x, y: x + y, 
            (f(image, self.pts_dist.sample(), self.params_dist.sample()) for f in self.sub_filters))

    def add_sub_filter(self, sub_filter):
        self.sub_filters.append(sub_filter)

class Sub_Filter:
    @staticmethod
    def apply(image, pts, params):
        return np.zeros(image.shape)

class Transparency(Sub_Filter):
    @staticmethod
    def rgba2rgb(img):
        return (1-img[:,:,3])[:,:,None] * img[:,:,:3] \
            + img[:,:,3][:,:,None]/255 * np.array([255,255,255])
    @staticmethod
    def apply(image, pts, params):
        h, w, _ = image.shape
        transp = np.random.random((h,w)) ** 3
        temp = np.concatenate([image, transp[:,:,None]], axis=2)
        temp = Transparency.rgba2rgb(temp).astype(np.uint8)
        return temp - image

class Gaussian_Noise(Sub_Filter):
    @staticmethod
    def gaussian_noise(img, mean=0, sigma=8):
        a, b, _ = img.shape
        gauss_noise = np.random.normal(mean, sigma, (a,b))
        new_watermark = np.zeros((img.shape), np.float32)
        new_watermark[:,:,0] = gauss_noise
        new_watermark[:,:,1] = gauss_noise
        new_watermark[:,:,2] = gauss_noise
        new_watermark = new_watermark.astype(np.uint8)
        return new_watermark
    @staticmethod
    def apply(image, pts, params):
        h, w, _ = image.shape
        image = np.zeros(image.shape, np.uint8)
        for i in range(1,20):
            p = (18 - i)/20
            center_y = int(h/2)
            center_x = int(w/2)
            top_y = center_y - int(h*p/2)
            left_x = center_x - int(w*p/2)
            bottom_y = top_y + int(h*p)
            right_x = left_x + int(w*p)
            f = image[top_y:bottom_y, left_x:right_x]
            new_f = Gaussian_Noise.gaussian_noise(f, 0, 0.25*i)
            image[top_y:bottom_y, left_x:right_x] = new_f
        return image

class Gaussian_Noise2(Sub_Filter):
    @staticmethod
    def gaussian_noise(img, mean=0, sigma=8):
        a, b, _ = img.shape
        new_watermark = np.zeros((img.shape), np.float32)
        new_watermark[:,:,0] = img[:,:,0] + np.random.normal(mean, sigma, (a,b))
        new_watermark[:,:,1] = img[:,:,1] + np.random.normal(mean, sigma, (a,b))
        new_watermark[:,:,2] = img[:,:,2] + np.random.normal(mean, sigma, (a,b))
        cv.normalize(new_watermark, new_watermark, 0, 255, cv.NORM_MINMAX, dtype=-1)
        new_watermark = new_watermark.astype(np.uint8)
        return new_watermark
    @staticmethod
    def apply(image, pts, params):
        h, w, _ = image.shape
        temp = image.copy()
        for i in range(1,20):
            p = (18 - i)/20
            center_y = int(h/2)
            center_x = int(w/2)
            top_y = center_y - int(h*p/2)
            left_x = center_x - int(w*p/2)
            bottom_y = top_y + int(h*p)
            right_x = left_x + int(w*p)
            f = image[top_y:bottom_y, left_x:right_x]
            new_f = Gaussian_Noise2.gaussian_noise(f, 0, 1.5*i)
            image[top_y:bottom_y, left_x:right_x] = new_f
        return image - temp

class Gaussian_Distribution(Distribution):
    def sample(self, img):
        h,w,_ = img.shape
        return np.random.multivariate_normal([h//2, w//2], [[h,0],[0,w]], 2).astype(np.int32)[1]
class Gaussian_Blur(Sub_Filter):
    @staticmethod
    def apply(image, pts, params):
        gd = Gaussian_Distribution()
        image = image.copy()
        x,y = pts
        r = 7
        image[x-3*r:x+3*r, y-3*r:y+3*r] = cv.GaussianBlur(image[x-3*r:x+3*r, y-3*r:y+3*r], (r,r), 1)
        return image

class FFT(Sub_Filter):
    @staticmethod
    def apply(image, pts, params):
        img_fft = np.fft.fft2(image)
        temp = img_fft + (np.random.random(img_fft.shape) - 0.5) * img_fft.std() * 0.25
        temp = np.abs(np.fft.ifft2(temp)).astype(np.uint8)
        return temp - image
