from abc import abstractmethod
import numpy as np
from menpo.visualize import print_dynamic
from menpo.shape import PointCloud
from copy import deepcopy

from ERT.tools import fit_shape_to_box
from Image_deal.PixelExtractor import RegressorBuilder, Regressor
from ERT import tools

# Here is the first part: the inner cascade.
# This is used to realize the boosting procedure for each step.
# And to be specifically, it is the to combine these trees together.

class InnerCascadeBuilder(RegressorBuilder):
    def __init__(self, n_primitive_regressors, primitive_builder, feature_extractor_builder):
        self.n_primitive_regressors = n_primitive_regressors
        self.primitive_builder = primitive_builder
        self.feature_extractor_builder = feature_extractor_builder

    # This one is left here to be used in the final builder as parent class.
    @abstractmethod
    def precompute(self, pixel_vectors, feature_extractor, mean_shape):
        return None

    @abstractmethod
    def post_process(self, primitive_regressors):
        return None

    def to_mean(self, shape):
        return tools.transform_to_mean_shape(shape, self.mean_shape)

    def build(self, images, targets, extra):
        shapes, mean_shape, i_stage = extra
        self.mean_shape = mean_shape
        assert(len(images) == len(shapes))

        feature_extractor = self.feature_extractor_builder.build(images, shapes, targets, (mean_shape, i_stage))

        # Extract shape-indexed pixels from images.
        pixel_vectors = np.array([feature_extractor.extract_features(img, shape, self.to_mean(shape).pseudoinverse())
                                  for (img, shape) in zip(images, shapes)])

        data = self.precompute(pixel_vectors, feature_extractor, mean_shape)

        primitive_regressors = []
        for i in range(self.n_primitive_regressors):
            print_dynamic("Building primitive regressor {}".format(i))
            primitive_regressor = self.primitive_builder.build(pixel_vectors, targets, data)
            # Update targets.
            targets -= [primitive_regressor.apply(pixel_vector, data) for pixel_vector in pixel_vectors]
            primitive_regressors.append(primitive_regressor)

        return InnerCascade((feature_extractor, primitive_regressors, mean_shape, self.post_process(primitive_regressors)))

class InnerCascade(Regressor):
    def __init__(self, data):
        feature_extractor, regressors, mean_shape, extra = data
        n_landmarks = mean_shape.n_points
        self.n_landmarks = n_landmarks
        self.feature_extractor = feature_extractor
        self.regressors = regressors
        self.mean_shape = mean_shape
        self.extra = extra

    def apply(self, image, shape):
        mean_to_shape = tools.transform_to_mean_shape(shape, self.mean_shape).pseudoinverse()
        shape_indexed_features = self.feature_extractor.extract_features(image, shape, mean_to_shape)
        res = PointCloud(np.zeros((self.n_landmarks, 2)), copy=False)
        for r in self.regressors:
            offset = r.apply(shape_indexed_features, self.extra)
            res.points += offset.reshape((self.n_landmarks, 2))
        return mean_to_shape.apply(res)

class RegressionForestBuilder(InnerCascadeBuilder):
    def __init__(self, n_trees, tree_builder, feature_extractor_builder):
        # This is for the use of InnerCascadeBuilder functions:
        super(self.__class__, self).__init__(n_trees, tree_builder, feature_extractor_builder)

    def precompute(self, pixel_vectors, pixel_extractor, mean_shape):
        pixel_coords = mean_shape.points[pixel_extractor.lmark] + pixel_extractor.pixel_coords
        return pixel_coords, mean_shape

# Here is used for outer cascade.
# I believe it is more like a forest.

class CascadedShapeRegressorBuilder(RegressorBuilder):
    def __init__(self, n_stages, n_perturbations, weak_builder):
        self.n_stages = n_stages
        self.weak_builder = weak_builder
        self.n_perturbations = n_perturbations
        self.n_landmarks = 0

    def build(self, images, gt_shapes, boxes):
        self.mean_shape = tools.centered_mean_shape(gt_shapes)
        self.n_landmarks = self.mean_shape.n_points
        # Generate initial shapes with perturbations.
        print_dynamic('Generating initial shapes')
        shapes = np.array([tools.fit_shape_to_box(self.mean_shape, box) for box in boxes])

        print_dynamic('Perturbing initial estimates')
        if self.n_perturbations > 1:
            images, shapes, gt_shapes, boxes = tools.perturb_shapes(images, shapes, gt_shapes, boxes,
                                                                   self.n_perturbations, mode='mean_shape')

        assert(len(boxes) == len(images))
        assert(len(shapes) == len(images))
        assert(len(gt_shapes) == len(images))

        print('\nSize of augmented dataset: {} images.\n'.format(len(images)))

        weak_regressors = []
        for j in range(self.n_stages):
            # Calculate normalized targets.
            deltas = [gt_shapes[i].points - shapes[i].points for i in range(len(images))]
            targets = np.array([tools.transform_to_mean_shape(shapes[i], self.mean_shape).apply(deltas[i]).reshape((2*self.n_landmarks,))
                                for i in range(len(images))])

            weak_regressor = self.weak_builder.build(images, targets, (shapes, self.mean_shape, j))
            # Update current estimates of shapes.
            for i in range(len(images)):
                offset = weak_regressor.apply(images[i], shapes[i])
                shapes[i].points += offset.points
            weak_regressors.append(weak_regressor)
            print("\nBuilt outer regressor {}\n".format(j))

        return CascadedShapeRegressor(self.n_landmarks, weak_regressors, self.mean_shape)


class CascadedShapeRegressor(Regressor):
    def __init__(self, n_landmarks, weak_regressors, mean_shape):
        self.n_landmarks = n_landmarks
        self.weak_regressors = weak_regressors
        self.mean_shape = mean_shape

    def apply(self, image, extra):
        boxes, init_num, initial_shapes = extra

        if initial_shapes is None:
            initial_shapes = np.array([fit_shape_to_box(self.mean_shape, box) for box in boxes])

        shapes = deepcopy(initial_shapes)

        for i, shape in enumerate(shapes):
            init_shapes = tools.perturb_init_shape(initial_shapes[i].copy(), init_num)
            for j in range(init_num):
                for r in self.weak_regressors:
                    offset = r.apply(image, init_shapes[j])
                    init_shapes[j].points += offset.points
            shape.points[:] = tools.get_median_shape(init_shapes).points

        return initial_shapes, shapes