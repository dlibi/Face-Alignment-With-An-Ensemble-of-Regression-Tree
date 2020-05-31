#import unittest
import sys
import os
import menpo.io as mio
from menpo.visualize import print_dynamic
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from ERT import tools
import ERT
import menpodetect

# This is a error function from others program based on the eye distance.
def compute_error(shape, gt_shape):
    eye_dis = np.linalg.norm(gt_shape.points[36]-gt_shape.points[45])
    return np.linalg.norm(shape.points-gt_shape.points, axis=1).mean()/eye_dis

def test_model(model, test_images, num_init):
    face_detector = menpodetect.dlib.load_dlib_frontal_face_detector()
    test_gt_shapes = tools.get_gt_shapes(test_images)
    test_boxes = tools.get_bounding_boxes(test_images, test_gt_shapes, face_detector)

    initial_errors = []
    final_errors = []

    initial_shapes = []
    final_shapes = []

    for k, (im, gt_shape, box) in enumerate(zip(test_images, test_gt_shapes, test_boxes)):
        init_shapes, fin_shapes = model.apply(im, ([box], num_init, None))

        init_shape = tools.get_median_shape(init_shapes)
        final_shape = fin_shapes[0]

        initial_shapes.append(init_shape)
        final_shapes.append(final_shape)

        initial_errors.append(compute_error(init_shape, gt_shape))
        final_errors.append(compute_error(final_shape, gt_shape))

        print_dynamic('{}/{}'.format(k + 1, len(test_images)))

    return initial_errors, final_errors, initial_shapes, final_shapes

def fit_all(model_builder, train_images, test_images, num_init):
    face_detector = menpodetect.dlib.load_dlib_frontal_face_detector()
    train_gt_shapes = tools.get_gt_shapes(train_images)
    train_boxes = tools.get_bounding_boxes(train_images, train_gt_shapes, face_detector)

    model = model_builder.build(train_images, train_gt_shapes, train_boxes)

    initial_errors, final_errors, initial_shapes, final_shapes = test_model(model, test_images, num_init)

    return initial_errors, final_errors, initial_shapes, final_shapes, model


# # Use two of the images to do both train and test.
# images = ['einstein.jpg', 'lenna.png']
# test_images = np.array([mio.import_builtin_asset(image).as_greyscale(mode='average').
#                        crop_to_landmarks_proportion_inplace(0.5) for image in images])
# train_images = test_images

def test_all(test, model_builder, test_images, train_images):
    initerr, finerr, _, _, _ = fit_all(model_builder, test_images, train_images, num_init=1)

    init_mean_error = np.mean(initerr)
    fin_mean_error = np.mean(finerr)

    print("Mean initial error: {}\n".format(init_mean_error))
    print("Mean final error: {}\n".format(fin_mean_error))

    test.failIfAlmostEqual(fin_mean_error, init_mean_error/5.0)
    test.failUnlessAlmostEqual(fin_mean_error, 0, places=4)

# class ERTTest(unittest.TestCase):
#     def test_end2end(self):
#         ert_builder = ERT.builder.ERTBuilder(n_stages=1, n_trees=1, MU=1, n_perturbations=1)
#         test_all(self, ert_builder, test_images, train_images)
#
# # def main():
# #     unittest.main()
#
# if __name__ == '__main__':
#     unittest.main()
