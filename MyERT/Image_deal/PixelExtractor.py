from abc import abstractmethod
import numpy as np
from ERT.tools import sample_image
# The class in this page is used to stand for the underlying class
# They are decorated with @abstractmethod so that when we use it
# the method in it has to be realized.

# The first two are used to extract pixels from given image and shapes
class FeatureExtractor(object):
    @abstractmethod
    def apply(self, image, shape):
        pass

class FeatureExtractorBuilder(object):
    @abstractmethod
    def build(self, images, shapes, targets, extra):
        pass

# This class use the former two as basic concepts to finally get the pixel vector list.
class PixelExtractor(FeatureExtractor):
    def __init__(self, n_landmarks, n_pixels, kappa, around_landmark):
        if around_landmark != -1:
            self.lmark = np.empty(n_pixels, dtype=np.int)
            self.lmark.fill(int(around_landmark))
        else:
            self.lmark = np.random.randint(low=0, high=n_landmarks, size=n_pixels)
        self.pixel_coords = np.random.uniform(low=-kappa, high=kappa, size=n_pixels*2).reshape(n_pixels, 2)

    def extract_features(self, img, shape, mean_to_shape):
        offsets = mean_to_shape.apply(self.pixel_coords)
        ret = shape.points[self.lmark] + offsets
        return sample_image(img, ret)


class PixelExtractorBuilder(FeatureExtractorBuilder):
    def __init__(self, n_landmarks, n_pixels, kappa, adaptive=False, around_landmark=-1):
        self.adaptive = adaptive
        self.n_landmarks = n_landmarks
        self.n_pixels = n_pixels
        self.kappa = kappa
        self.around_landmark = around_landmark

    def build(self, images, shapes, targets, extra):
        mean_shape, i_stage = extra
        kappa = self.kappa
        if self.adaptive:
            kappa -= i_stage*0.002
        return PixelExtractor(self.n_landmarks, self.n_pixels, kappa, around_landmark=self.around_landmark)



class Regressor(object):
    def apply(self, features, extra):
        pass

class RegressorBuilder(object):
    @abstractmethod
    def build(self, features, targets, extra):
        pass