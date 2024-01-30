import cv2
import numpy as np
import matplotlib.pyplot as plt

import model3d


BIN, OVERLAP = 6, 0.1


def init_points3D(dims):
    points3D = np.zeros((8, 3))
    cnt = 0
    for i in [1, -1]:
        for j in [1, -1]:
            for k in [1, -1]:
                points3D[cnt] = dims[[1, 0, 2]].T / 2.0 * [i, k, j * i]
                cnt += 1
    return points3D


def gen_3D_box(yaw, dims, cam_to_img, box_2D):
    dims = dims.reshape((-1, 1))
    box_2D = box_2D.reshape((-1, 1))
    points3D = init_points3D(dims)

    # Here the rotation is done around the Y axis. Just a convention in the code.
    rot_M = np.asarray([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    center = compute_center(points3D, rot_M, cam_to_img, box_2D, inds)

    points2D = points3D_to_2D(points3D, center, rot_M, cam_to_img)

    return points2D


def compute_center(points3D, rot_M, cam_to_img, box_2D, inds):
    fx = cam_to_img[0][0]
    fy = cam_to_img[1][1]
    u0 = cam_to_img[0][2]
    v0 = cam_to_img[1][2]

    W = np.array([[fx, 0, u0 - box_2D[0][0]],
                  [fx, 0, u0 - box_2D[2][0]],
                  [0, fy, v0 - box_2D[1][0]],
                  [0, fy, v0 - box_2D[3][0]]], dtype='float')
    center = None
    error_min = 1e10

    for ind in inds:
        y = np.zeros((4, 1))
        for i in range(len(ind)):
            RP = np.dot(rot_M, (points3D[ind[i]]).reshape((-1, 1)))
            y[i] = box_2D[i] * cam_to_img[2][3] - np.dot(W[i], RP) - cam_to_img[i // 2][3]

        result = solve_least_squre(W, y)
        error = compute_error(points3D, result, rot_M, cam_to_img, box_2D)

        if error < error_min and result[2, 0] > 0:
            center = result
            error_min = error

    return center


def draw_3D_box(image, points):
    points = points.astype(int)

    for i in range(4):
        point_1_ = points[2 * i]
        point_2_ = points[2 * i + 1]
        cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0, 255, 0), 1)

    # The red X at the front.
    cv2.line(image, tuple(points[0]), tuple(points[7]), (0, 0, 255), 2)
    cv2.line(image, tuple(points[1]), tuple(points[6]), (0, 0, 255), 2)

    for i in range(8):
        point_1_ = points[i]
        point_2_ = points[(i + 2) % 8]
        cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0, 255, 0), 1)

    return image


def solve_least_squre(W, y):
    U, Sigma, VT = np.linalg.svd(W)
    result = np.dot(np.dot(np.dot(VT.T, np.linalg.pinv(np.eye(4, 3) * Sigma)), U.T), y)
    return result


def points3D_to_2D(points3D, center, rot_M, cam_to_img):
    # General formula is: [2D] = K[R T][3D]
    # So for each 3D point, apply rotation, add translation (3D center)
    # and multiply K. At last, you will have a 3x1 vector, which you should normalize
    # by the third element (Homogenous coordinates).
    points2D = []
    for point3D in points3D:
        point3D = point3D.reshape((-1, 1))
        point = center + np.dot(rot_M, point3D)
        point = np.append(point, 1)
        point = np.dot(cam_to_img, point)
        point2D = point[:2] / point[2]
        points2D.append(point2D)
    points2D = np.asarray(points2D)

    return points2D


def compute_error(points3D, center, rot_M, cam_to_img, box_2D):
    # Get all of 8 corners from 3D box projected on image.
    points2D = points3D_to_2D(points3D, center, rot_M, cam_to_img)

    # Get a new bounding box from the 8 projected coreners.
    new_box_2D = np.asarray([np.min(points2D[:, 0]),
                             np.max(points2D[:, 0]),
                             np.min(points2D[:, 1]),
                             np.max(points2D[:, 1])]).reshape((-1, 1))

    # Sum the absolute difference of xmin, xmax, ymin, ymax for the 2D bbox,
    # and the new bbox from 8 projections.
    error = np.sum(np.abs(new_box_2D - box_2D))
    return error


# These will be used in the next cell
inds = []
indx = [1, 3, 5, 7]
indy = [0, 1, 2, 3]
for i in indx:
    for j in indx:
        for m in indy:
            for n in indy:
                inds.append([i, j, m, n])

# Dimension averages collected from the dataset
dims_avg = {'Car': [1.52130159, 1.64441129, 3.85729945],
            'Truck': [3.07044968, 2.62877944, 11.17126338],
            'Van': [2.18560847, 1.91077601, 5.08042328],
            'Tram': [3.56005102, 2.4002551, 18.52173469]}


# Plot the sample
test_image = cv2.imread('images/vlcsnap-2024-01-30-10h19m09s567.png')
test_image = cv2.resize(test_image, (1920, 1080))

# roi = cv2.selectROI("Select ROI", test_image, True, False)
# xmin, ymin, xmax, ymax = roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]
# print(xmin, ymin, xmax, ymax)
xmin, ymin, xmax, ymax = 1090, 182, 1505, 633

image_plot = test_image[:, :, ::-1]
plt.imshow(image_plot / 255.)
plt.show()

# If you are want to do a quick test, you can insert bounding box annotations manually.
final_boxes = [[xmin, ymin, xmax, ymax]]  # Only one box of course.

# Plot the patch
patch = image_plot[ymin:ymax, xmin:xmax]
plt.figure()
plt.imshow(patch / 255.)
plt.show()

model = model3d.build_model(input_shape=(224, 224, 3), weights='imagenet', freeze=False,
                            feature_extractor="mobilenetv2")
model.load_weights('models/checkpoints_weights_mobilenetv2.hdf5')

# #Just for quick tests
# calib_mat = [[7.215377e+02, 0.000000e+00, 9.60e+02, 4.485728e+01],
#              [0.000000e+00, 7.215377e+02, 5.40e+02, 2.163791e-01],
#              [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]]
calib_mat = [[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
            [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]]

final_image = test_image.copy()
for index, box in enumerate(final_boxes):
    xmin, ymin, xmax, ymax = box
    # Crop
    patch = test_image[ymin:ymax, xmin:xmax]
    # Resize
    patch = cv2.resize(patch, (224, 224))
    # Set the input pixels to be within (-0.5 , 0.5).
    patch = patch / 255.0 - 0.5

    patch = np.expand_dims(patch, axis=0)
    prediction = model.predict(patch)

    # Get the (cos, sin) of the bin with highest probabilitys
    max_anc = np.argmax(prediction[2][0])
    anchors = prediction[1][0][max_anc]

    # anchors=(cos, sin)
    if anchors[1] > 0:
        angle_offset = np.arccos(anchors[0])
    else:
        angle_offset = -np.arccos(anchors[0])

    wedge = 2. * np.pi / BIN
    theta_loc = angle_offset + max_anc * wedge

    fx = calib_mat[0][0]
    u0 = calib_mat[0][2]
    v0 = calib_mat[1][2]

    # As suggested in the original paper, we estimate the raye to 2D box center instead of
    # object's 3D center. Of course, it is not entirely accurate, but it is sufficient.
    box2d_center_x = (xmin + xmax) / 2.0
    theta_ray = np.arctan(fx / (box2d_center_x - u0))

    # Arctan's output is (-pi/2, pi/2). But theta_ray should be (0, pi).
    # From 0 to pi/2 the values are ok but if the object's center is on the left half,
    # the arctan will result in a negative theta_ray. This can be fixed by adding pi.
    if theta_ray < 0:
        theta_ray = theta_ray + np.pi

    # Final theta
    theta = theta_loc + theta_ray
    yaw = np.pi / 2 - theta

    # Here we use average dims of cars. If you use a trained model for 2D detection with
    # correct classes, you can simply replace this line. The training steps of 2D network is
    # provided at the end of the notebook.
    dims = dims_avg['Car'] + prediction[0][0]

    box_2D = np.asarray([box[1], box[0], box[3], box[2]], dtype=np.float32)
    points2D = gen_3D_box(yaw, dims, calib_mat, box_2D)  # switched yaw -> theta
    final_image = draw_3D_box(final_image, points2D)

final_image_show = final_image[:, :, ::-1]
plt.figure(figsize=(20, 20))
plt.imshow(final_image_show / 255.)
plt.show()
