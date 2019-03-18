import argparse
import cv2
import matplotlib.pyplot as plt
from face_match import FaceMatch
import time

parser = argparse.ArgumentParser()
parser.add_argument("--img1", type=str, required=True)
parser.add_argument("--img2", type=str, required=True)
args = parser.parse_args()


def sample_demo(img_base, img_target, face_match):
    # plt.subplot(511), plt.imshow(img2), plt.title('Original')
    # plt.xticks([]), plt.yticks([])

    distances_found = []

    for i in range(1, 5):
        img_target_blur = cv2.blur(img_target, (5 + i * 2, 5 + i * 2))

        my_distance = face_match.compare_faces(img_base, img_target_blur)
        distances_found.append(my_distance)
        # temp_f = round(my_distance, 2)
        # plt.subplot(511 + i), plt.imshow(img_target_blur), plt.title('{} Blurred {}x{}'.format(temp_f, i + 5 * 2, i + 5 * 2))
        # plt.xticks([]), plt.yticks([])


def box_avg_experiment(img_base, img_target, face_match, box_max_size=40):

    distances_found = []
    box_sizes = []

    for i in range(1, box_max_size):
        box_sizes.append(5 + i)
        img_target_blur = cv2.blur(img_target, (5 + i, 5 + i))
        distance = face_match.compare_faces(img_base, img_target_blur)
        print("Box size AVG experiment - Iteration {} box size {} distance {}".format(i, i+5, distance))
        if distance == -1:
            print("Face detection/align failed")
            break
        distances_found.append(distance)

    plt.plot(box_sizes, distances_found)
    plt.xlabel('Blur avg box size')
    plt.ylabel('Euclidean Distance')

    print(distances_found)

    return box_sizes, distances_found


fm = FaceMatch()

#img1 = cv2.imread(args.img1)
#img2 = cv2.imread(args.img2)

img1 = cv2.imread("./images/daniel-radcliffe.jpg")
img2 = cv2.imread("./images/daniel-radcliffe_2.jpg")

start_time = time.time()
face_match_distance = fm.compare_faces(img1, img2)
print("--- %s seconds ---" % (time.time() - start_time))

face_match_threshold = 1.10

print("distance = " + str(face_match_distance))
print("Result = " + ("same person" if face_match_distance <= face_match_threshold else "not same person"))


sample_demo(img1, img2, fm)
plt.show()
box_avg_experiment(img1, img2, fm)
plt.show()

# fm_new = FaceMatch("20180402-114759/20180402-114759.pb")
# face_match_distance = fm_new.compare_faces(img1, img2)
#
# sample_demo(img1, img2, fm_new)
# plt.show()
# box_avg_experiment(img1, img2, fm_new)
# plt.show()

