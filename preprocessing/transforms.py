import cv2
import numpy as np
import pywt
import random
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin

class Normalize(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.astype('float32') / 255.0

class DenoiseWavelet(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db1', level=1, thresholding='soft'):
        self.wavelet = wavelet
        self.level = level
        self.thresholding = thresholding

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        coeffs = pywt.wavedec2(X, self.wavelet, level=self.level)
        sigma = (1/0.6745) * np.median(np.abs(coeffs[-self.level] - np.median(coeffs[-self.level])))
        uthresh = sigma * np.sqrt(2 * np.log(X.size))
        coeffs = list(coeffs)
        coeffs[1:] = [(pywt.threshold(c[0], value=uthresh, mode=self.thresholding),
                       pywt.threshold(c[1], value=uthresh, mode=self.thresholding),
                       pywt.threshold(c[2], value=uthresh, mode=self.thresholding)) for c in coeffs[1:]]
        return pywt.waverec2(coeffs, self.wavelet)

class GaussianBlur(BaseEstimator, TransformerMixin):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return cv2.GaussianBlur(X, (self.kernel_size, self.kernel_size), 0)

class ApplyHOG(BaseEstimator, TransformerMixin):
    def __init__(self, visualize=False):
        self.visualize = visualize

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.visualize:
            hog_features, hog_image = hog(X, visualize=self.visualize)
            return hog_image
        else:
            hog_features = hog(X, visualize=self.visualize)
            return hog_features
        
class Clahe(BaseEstimator, TransformerMixin):
    def __init__(self, clipLimit):
        self.clipLimit = clipLimit
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit)
        return clahe.apply(X)
    
class LRandomFlip(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.astype(np.uint8)
        return np.fliplr(X)

class URandomFlip(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.astype(np.uint8)
        return np.flipud(X)

class D1RandomFlip(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.astype(np.uint8)
        return np.transpose(X)

class D2RandomFlip(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.astype(np.uint8)
        return np.fliplr(np.flipud(np.transpose(X)))
