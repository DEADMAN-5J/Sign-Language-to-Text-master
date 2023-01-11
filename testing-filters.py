import cv2

# camera = cv2.VideoCapture(0)

# while True:
#     _, frame = camera.read()
#     #cv2.imshow("Camera", frame)

#     # edges = cv2.Canny(frame, 120, 160)
#     #edges = auto_canny(frame)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray,(5,5),2)
#     cv2.imshow("Canny", blur)

#     if cv2.waitKey(5) == ord('x'):
#         break

  
originalImage = cv2.imread('signs-ISL.png')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(grayImage,(5,5),2)
  

# cv2.imshow('Original image',originalImage)
cv2.imshow('Gray image', blur)
  
cv2.waitKey(0)
cv2.destroyAllWindows()

