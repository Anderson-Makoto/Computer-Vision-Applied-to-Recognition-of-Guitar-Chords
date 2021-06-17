import cv2

img = cv2.imread("./scripts/deep_learning_segmentation/DO/2.jpg", 0)
print(img)
img = cv2.resize(img, (341, 256))
cv2.imwrite("./2.jpg", img)