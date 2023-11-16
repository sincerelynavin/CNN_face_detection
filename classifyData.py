import cv2


def get_coordinate(event,x,y,flags,param):
    print(f'{x},{y}')
    # if event == cv2.EVENT_LBUTTONDBLCLK:
        # cv2.circle(img,(x,y),100,(255,0,0),-1)
        # mouseX,mouseY = x,y

image_path = "./testData/facetest8.jpg"
# Read the input image
img = cv2.imread(image_path)

# Draw rectangle around the faces
x, y, w, h = 241,251,129,129

cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.namedWindow('img')
cv2.imshow('img', img)
cv2.setMouseCallback('img',get_coordinate)

cv2.waitKey()