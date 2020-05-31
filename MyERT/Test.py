import numpy as np
import hickle as hkl
import random
from menpofit.visualize import plot_cumulative_error_distribution
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from copy import deepcopy

from ERT.builder import ERTBuilder
from TestFunc import fit_all, test_model
from ERT import tools

import site
site.addsitedir('..')

MODEL_NAME = "ert_300m.hkl"

# Prepare the data for train and test
# indoor_train = tools.read_images("../MyERT/Images/01_Indoor/Indoor_Train/", normalise=True)
indoor_test = tools.read_images("../MyERT/Images/01_Indoor/Indoor_Test/", normalise=True)
# outdoor_train = tools.read_images("../MyERT/Images/02_Outdoor/Outdoor_Train/", normalise=True)
outdoor_test = tools.read_images("../MyERT/Images/02_Outdoor/Outdoor_Test/", normalise=True)
#
# train_images = np.concatenate([indoor_train, outdoor_train])
test_images = np.concatenate([indoor_test, outdoor_test])
#
# # Build the model and get the fit results
# builder = ERTBuilder(n_stages=10, n_trees=500, MU=0.1, n_perturbations=20)
# initial_errors, final_errors, initial_shapes, final_shapes, model = fit_all(builder, train_images, test_images, num_init=1)
#
# hkl.dump(model, "../MyERT/Models/"+MODEL_NAME)
#
# print("Mean initial error: {}".format(np.mean(initial_errors)))
# print("Mean final error: {}".format(np.mean(final_errors)))
#
# plot_cumulative_error_distribution(final_errors)


model = hkl.load("../MyERT/Models/"+MODEL_NAME)
print("model loaded")
initial_errors, final_errors, initial_shapes, final_shapes = test_model(model, test_images, num_init=5)

print("Mean initial error: {}".format(np.mean(initial_errors)))
print("Mean final error: {}".format(np.mean(final_errors)))

plot_cumulative_error_distribution(final_errors)


worst = np.argmax(final_errors)
print(worst)

initial_errors, final_errors, initial_shapes, final_shapes = test_model(model, test_images, num_init=1)

print("Mean initial error: {}".format(np.mean(initial_errors)))
print("Mean final error: {}".format(np.mean(final_errors)))

plot_cumulative_error_distribution(final_errors)


initial_errors, final_errors, initial_shapes, final_shapes = test_model(model, test_images, num_init=5)
print("Mean initial error: {}".format(np.mean(initial_errors)))
print("Mean final error: {}".format(np.mean(final_errors)))

plot_cumulative_error_distribution(final_errors)
plt.show()

ii = random.randint(0, 67)
test_images[ii].landmarks['dlib_0'].points = final_shapes[ii].points
test_images[ii].view_landmarks(marker_face_colour='y', marker_edge_colour='y', group='dlib_0')


