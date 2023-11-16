import cv2
import json
import os


def intersect_square(tl1, br1, tl2, br2):
    # Calculate the intersection coordinates
    x1 = max(tl1[0], tl2[0])
    y1 = max(tl1[1], tl2[1])
    x2 = min(br1[0], br2[0])
    y2 = min(br1[1], br2[1])

    # Check if the intersection is valid
    if (
        x1 < x2 and y1 < y2):
        return ((x1, y1), (x2, y2))
    else:
        return None

# Config
window_size = 200
stride_size = 20

# Path to test data
datasetPath = './testData'
datasetOutputPath = './trainingData'

# load test config
with open(f'./{datasetPath}/groundTruth.json') as json_data:
    groundTruth = json.load(json_data)

# print(groundTruth)
# Iterate over test images

training_ground_truth = []

imgID=0
for test_image_config in groundTruth:

    image_path = os.path.join(datasetPath,test_image_config.get('fileName'))
    faces_in_image = test_image_config.get('faces')

    print(image_path)
    # Read the input image
    img = cv2.imread(image_path)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Split image into sub images
    for x in range(0, img.shape[1]-window_size, stride_size):
        for y in range(0, img.shape[0]-window_size, stride_size):
            sub_image = img[y:y+window_size, x:x+window_size]
            sub_image_has_face = 0
            sub_image_name = f'{imgID}.jpg'
            # check if face is in sub image
            for face in faces_in_image:
                if intersect_square((x, y), (x+window_size, y+window_size), (face[0], face[1]), (face[2], face[3])):
                    sub_image_has_face = 1

            cv2.imshow('img', sub_image)
            cv2.imwrite(os.path.join(datasetOutputPath, sub_image_name), sub_image)
            # cv2.waitKey()

            training_ground_truth.append({ "fileName": sub_image_name, "has_face": sub_image_has_face })

            # increment image ID
            imgID =  imgID+1
#

# write json to file
json.dump(training_ground_truth, open(os.path.join(datasetOutputPath, 'groundTruth.json'), 'w+'))


