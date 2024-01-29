import os
import copy
import argparse

import cv2
import numpy as np

import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2

from clearml import Task
from clearml import Dataset


parser = argparse.ArgumentParser()
parser.add_argument('-remotely', default=True, action='store_true', help='Train remotely')
parser.add_argument('-epochs', default=150, help='Total number of epochs')
parser.add_argument('-batch', default=32, help='Batch size')
parser.add_argument('-feature-extractor', default='mobilenetv2', help='Feature extractor (vgg16 or mobilenetv2)',
                    choices=['vgg16', 'mobilenetv2'])
args = parser.parse_args()


task = Task.init(
    project_name='Radius/KITTI-3D',
    task_name=f'baseline {args.feature_extractor} {args.epochs} epochs',
    output_uri=True
)

if args.remotely:
    task.execute_remotely(
        queue_name='default',
        clone=False,
        exit_process=True
    )

data_path = Dataset.get(dataset_name='KITTI-3D').get_local_copy()


# This is for the orientation estimation.
BIN, OVERLAP = 6, 0.1

W = 1.
# The crop shapes for the input 3D network.
NORM_H, NORM_W = 224, 224

# Maximum bounding box jitter for data augmentation while training 3D network
MAX_JIT = 5

# Classes to be considered and extracted from kitti dataset, while training the 3D network
VEHICLES = ['Car', 'Truck', 'Van']

# These are used to filter out useless and bad objects from dataset (for training)
MAX_OCCLUSION = 1  # anything <= will be included.
MAX_TRUNCATION = 0.8  # anything < will be included.

# Path to the images and labels dir.
image_dir = os.path.join(data_path, 'training/image_2/')
label_dir = os.path.join(data_path, 'training/label_2/')

# ----------------------
# ------- UTILS --------
# ----------------------

def compute_anchors(angle):
    anchors = []

    wedge = 2. * np.pi / BIN  # Length of each bin in Radian.

    # Each angle will lie somewhere in a bin. But we need to find the closest. That is, given a bin, we want to know which bound is closer to our angle.
    # For instance if our bin is (pi/2, pi), we want to know whether the angle is easier to reference from pi/2 or pi.
    # So we keep both lower and upper bound using l_index and r_index, respectively.
    l_index = int(angle / wedge)
    r_index = l_index + 1

    # Now we check if our angle is more closer to current bin's lower bound, or its upper bound.

    # If close enough to the lower bound, consider current index..
    if (angle - l_index * wedge) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([l_index, angle - l_index * wedge])
    # If close enough to upper bound,
    if (r_index * wedge - angle) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([r_index % BIN, angle - r_index * wedge])

    return anchors


def prepare_input_and_output(train_inst):
    # Read the image.
    img = cv2.imread(image_dir + train_inst['image'])

    # Read AND jitter each bounding box
    xmin = train_inst['xmin'] + np.random.randint(-MAX_JIT, MAX_JIT + 1)
    ymin = train_inst['ymin'] + np.random.randint(-MAX_JIT, MAX_JIT + 1)
    xmax = train_inst['xmax'] + np.random.randint(-MAX_JIT, MAX_JIT + 1)
    ymax = train_inst['ymax'] + np.random.randint(-MAX_JIT, MAX_JIT + 1)

    # Ensure that your coordinates are withing ranges after jittering.
    shape = img.shape
    xmin = max(min(xmin, shape[1] - 1), 0)
    xmax = max(min(xmax, shape[1] - 1), 0)
    ymin = max(min(ymin, shape[0] - 1), 0)
    ymax = max(min(ymax, shape[0] - 1), 0)

    # Crop the frame.
    img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1]).astype(np.float32)

    # Randomly decide whether to flip or not.
    flip = np.random.binomial(1, .5)
    if flip > 0.5: img = cv2.flip(img, 1)

    # resize the patch to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))

    # Make pixel values [-0.5 to 0.5].
    img = img / 255.0 - 0.5

    # Return data but take care of returining proper orientation (will change if image is flipped).
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']


def data_gen(all_objs, batch_size):
    # Total objects (patches) that we have
    num_obj = len(all_objs)

    keys = range(num_obj)
    np.random.shuffle(list(keys))

    # For each batch, we will have indices of [ l_bound ...... r_bound ) (not including r_bound itself).
    # Usually r_bound - l_bound  should be equal to batch size.
    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj

    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(list(keys))

        currt_inst = 0
        x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))  # Image batch
        d_batch = np.zeros((r_bound - l_bound, 3))  # Dimension batch
        o_batch = np.zeros((r_bound - l_bound, BIN, 2))  # orientation batch
        c_batch = np.zeros((r_bound - l_bound, BIN))  # confidences batch

        # Iterate the batch
        for key in keys[l_bound:r_bound]:
            # Prepare the data for the current frame.
            image, dimension, orientation, confidence = prepare_input_and_output(all_objs[key])

            x_batch[currt_inst, :] = image
            d_batch[currt_inst, :] = dimension
            o_batch[currt_inst, :] = orientation
            c_batch[currt_inst, :] = confidence

            currt_inst += 1

        # Yield the prepared batch.
        yield x_batch, [d_batch, o_batch, c_batch]

        # Go for the next batch
        l_bound = r_bound
        r_bound = r_bound + batch_size

        # Limit the r_bount to max valid index.
        if r_bound > num_obj: r_bound = num_obj


def l2_normalize(x):
    # Compute the second norm for each (sin, cos) pare and normalize the values.
    # So (sin, cos) will be normalized into ( sin/sqrt(sin^2+cos^2) , cos/sqrt(sin^2+cos^2)).
    # Thus if the network gives (a,b), we are always sure that a^2 + b^2 = 1 and we can use arctan with no worries.
    return tf.nn.l2_normalize(x, axis=2)


def orientation_loss(y_true, y_pred):
    # Here we have two 3D arrays with shape (batch_size, bin count, 2). the 2 is for the (sin, cos) vector.
    # The loss value, however, should be a scalar.

    # Make (sin, cos) into (sin^2, cos^2) and then sum them up into 1 scalar.
    # So the shape is now (batch_size, bin_count)
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)

    # Not sure about this line. I believe it assigns true for every bin that
    # has enough overlap with the true angle. Because, in the "Process data" cell,
    # we inserted (0,0) for other bins and only the one or two bins with enough overlap received the (sin, cos)
    anchors = tf.greater(anchors, tf.constant(0.5))

    # Now for each row, sum the values. So the shape is now (batch size).
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

    # We use cosine similarity for the loss. we compute  cos(alpha)= a.b / |a||b|.
    # alpha is the angle between ground truth vector and the estimation.
    # The ideal value would be cos(0) = 1. But gradient decent tries to MINIMIZE the loss.
    # So we add a - behind it. Now the ideal loss value is -1 and the network gets trained in the right direction.
    loss = -(y_true[:, :, 0] * y_pred[:, :, 0] + y_true[:, :, 1] * y_pred[:, :, 1])

    # For each batch, sum all loss values for all bins. So the shape becomes (batch size)
    loss = tf.reduce_sum(loss, axis=1)

    # Now normalize the loss. If I've got it right, the anchors are an array of (bin count, bin count, ...)
    # So now the loss becomes (loss0/relevant_bin_count, loss1/relevant_bin_count, ...) with shape (batch size).
    loss = loss / anchors

    # Use mean to turn the vector loss into scalar.
    return tf.reduce_mean(loss)


# -------------------------------------
# ------- ANNOTATIONS AND DATA --------
# -------------------------------------

def parse_annotation(label_dir, image_dir):
    # Here we prepare all objects (patches), their attributes, and the
    # average dimension across the dataset.
    all_objs = []

    # The average vector and the object count used for averaging.
    dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key: 0 for key in VEHICLES}

    # Iterate through label files.
    for label_file in os.listdir(label_dir):

        # If you are using another dataset, take care of this.
        image_file = label_file.replace('txt', 'png')

        # Iterate through lines in each label file
        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            # Each row will have this structure:
            # Class Truncated Occluded Theta(local) Xmin Ymin Xmax Ymax Dim Dim Dim T T T 
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            # Make sure it is a relevant class and with enough visibility (based on global parameters)
            if line[0] in VEHICLES and truncated < MAX_TRUNCATION and occluded <= MAX_OCCLUSION:
                # This is pretty confusing here. I will explain it somewhere in the documentation. Sorry :(
                new_alpha = -float(line[3]) + 3 * np.pi / 2
                new_alpha = new_alpha - np.floor(new_alpha / (2. * np.pi)) * (2. * np.pi)

                obj = {'name': line[0],
                       'image': image_file,
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha,
                       # The next 3 are not used for training the network, but for benchmarking translation vector accuracy. 
                       'translation': np.array([float(number) for number in line[11:14]]),
                       'truncated': truncated,
                       'occluded': occluded
                       }

                # Update the average while reading the new object's dimensions
                dims_avg[obj['name']] = dims_cnt[obj['name']] * dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                # Add the object to the list
                all_objs.append(obj)

    # return the object list and the average dimensions.
    return all_objs, dims_avg


all_objs, dims_avg = parse_annotation(label_dir, image_dir)

print("number of objects: {}".format(len(all_objs)))
print(" ----- \n A sample object(patch):\n {} \n -----".format(all_objs[0]))
print("The average dimensions: {}".format(dims_avg))

for obj in all_objs:

    # Fix dimensions, compute the residual
    obj['dims'] = obj['dims'] - dims_avg[obj['name']]

    # Make all residuals 0
    orientation = np.zeros((BIN, 2))
    # Make all bin confidences 0
    confidence = np.zeros(BIN)

    # turn angles into -> [bin# , residual]
    anchors = compute_anchors(obj['new_alpha'])

    for anchor in anchors:
        # compute the cosine and sine of the residual
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])

        # make the confidence of relative bin# 1
        confidence[anchor[0]] = 1.

    # normalize confidences
    confidence = confidence / np.sum(confidence)

    # add the computed orientation and confidence to the obj
    obj['orient'] = orientation
    obj['conf'] = confidence

    # Fix orientation and confidence for flip
    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)

    anchors = compute_anchors(2. * np.pi - obj['new_alpha'])

    for anchor in anchors:
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1

    confidence = confidence / np.sum(confidence)

    obj['orient_flipped'] = orientation
    obj['conf_flipped'] = confidence


# -------------------------
# ------- 3D MODEL --------
# -------------------------
# Construct the network
def build_model(input_shape=(224, 224, 3), weights=None, freeze=False, feature_extractor='vgg16'):
    if feature_extractor == 'mobilenetv2':
        feature_extractor_model = MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)
    elif feature_extractor == 'vgg16':
        feature_extractor_model = VGG16(include_top=False, weights=weights, input_shape=input_shape)
    else:
        print(
            "Requested a non-existing feature extractor model. Either choose from mobilenetv2 and vgg16 or add your own to the code")
        exit(-1)

    if freeze:
        for layer in feature_extractor_model.layers:
            layer.trainable = False

    x = layers.Flatten()(feature_extractor_model.output)

    dimension = layers.Dense(512)(x)
    dimension = layers.LeakyReLU(alpha=0.1)(dimension)
    dimension = layers.Dropout(0.5)(dimension)
    dimension = layers.Dense(3)(dimension)
    dimension = layers.LeakyReLU(alpha=0.1, name='dimension')(dimension)

    orientation = layers.Dense(256)(x)
    orientation = layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = layers.Dropout(0.5)(orientation)
    orientation = layers.Dense(BIN * 2)(orientation)
    orientation = layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = layers.Reshape((BIN, -1))(orientation)
    orientation = layers.Lambda(l2_normalize, name='orientation')(orientation)

    confidence = layers.Dense(256)(x)
    confidence = layers.LeakyReLU(alpha=0.1)(confidence)
    confidence = layers.Dropout(0.5)(confidence)
    confidence = layers.Dense(BIN, activation='softmax', name='confidence')(confidence)

    model = Model(feature_extractor_model.input, outputs=[dimension, orientation, confidence])
    model.summary()

    return model


# You can replace 'mobilenet' with 'vgg16' to change the feature extractor implementation in build_model().
model = build_model(input_shape=(224, 224, 3), weights='imagenet', freeze=False, feature_extractor=args.feature_extractor)

os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint(f'/models/checkpoints_weights_{args.feature_extractor}.hdf5', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', period=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
tensorboard = TensorBoard(log_dir='/logs/', histogram_freq=0, write_graph=True, write_images=False)

all_exams = len(all_objs)
trv_split = int(0.85 * all_exams)
batch_size = args.batch

# As I understood, Kitti's 2D Object dataset is already shuffeled. So there is no need to shuffle the dataset.
# Nevertheless, you can do so with next line: 
# np.random.shuffle(all_objs)

train_gen = data_gen(all_objs[:trv_split], batch_size)
valid_gen = data_gen(all_objs[trv_split:all_exams], batch_size)

train_num = int(np.ceil(trv_split / batch_size))
valid_num = int(np.ceil((all_exams - trv_split) / batch_size))

print("training configurations:")
print("all data:", all_exams)
print("training data:", trv_split)
print("batch size:", batch_size)
print("train_num", train_num)
print("valid_num", valid_num)

minimizer = Adam(lr=1e-5)
model.compile(optimizer=minimizer,  # minimizer,
              loss={'dimension': 'mean_squared_error', 'orientation': orientation_loss,
                    'confidence': 'categorical_crossentropy'},
              loss_weights={'dimension': 2., 'orientation': 1., 'confidence': 4.}, metrics=['mse', 'mae'])

model.fit_generator(generator=train_gen,
                    steps_per_epoch=train_num,
                    epochs=args.epochs,
                    verbose=1,
                    callbacks=[early_stop, checkpoint, tensorboard],
                    validation_data=valid_gen,
                    validation_steps=valid_num,
                    )

model.save(f'/models/weights_{args.feature_extractor}.hdf5')
