import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import argparse

import matplotlib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--img1", type = str, required=True)
parser.add_argument("--img2", type = str, required=True)
args = parser.parse_args()

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]


def getFace(img, nn_image_size):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (nn_image_size, nn_image_size), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]], 'embedding':getEmbedding(prewhitened, nn_image_size)})
    return faces


def getEmbedding(resized, nn_image_size):
    reshaped = resized.reshape(-1, nn_image_size, nn_image_size, 3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def compare2face(img1, img2, nn_image_size):
    face1 = getFace(img1, nn_image_size)
    face2 = getFace(img2, nn_image_size)
    if face1 and face2:
        # calculate Euclidean distance
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        return dist
    else:
        return -1


img1 = cv2.imread(args.img1)
img2 = cv2.imread(args.img2)
face_match_distance = compare2face(img1, img2, 160)
face_match_threshold = 1.10

print("distance = " + str(face_match_distance))
print("Result = " + ("same person" if face_match_distance <= face_match_threshold else "not same person"))

plt.subplot(511), plt.imshow(img2), plt.title('Original')
plt.xticks([]), plt.yticks([])

distances_found = []

for i in range(1, 5):
    img2_blur = cv2.blur(img2, (5+i*2, 5+i*2))

    my_distance = compare2face(img1, img2_blur)
    distances_found.append(my_distance)
    temp_f = round(my_distance, 2)
    plt.subplot(511+i), plt.imshow(img2_blur), plt.title('{} Blurred {}x{}'.format(temp_f, i+5*2, i+5*2))
    plt.xticks([]), plt.yticks([])

plt.show()

distances_found = []
box_sizes = []

for i in range(1, 10):

    box_sizes.append(i)
    my_distance = compare2face(img1, img2, i)
    distances_found.append(my_distance)

plt.plot(box_sizes, distances_found)
plt.xlabel('Size factor')
plt.ylabel('Euclidean Distance')
plt.show()

distances_found = []
box_sizes = []

for i in range(1, 40):

    box_sizes.append(5+i)
    img2_blur = cv2.blur(img2, (5+i, 5+i))
    my_distance = compare2face(img1, img2_blur)
    distances_found.append(my_distance)

plt.plot(box_sizes, distances_found)
plt.xlabel('Blur avg box size')
plt.ylabel('Euclidean Distance')

print(distances_found)
plt.show()
