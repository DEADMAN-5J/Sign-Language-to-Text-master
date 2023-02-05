from keras.models import model_from_json, load_model
# import keras.utils as image
# from keras.preprocessing import image
# from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# model = model_from_json('model-bw.json')
model = load_model('sg-model')

dirs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

model.summary()

# test_image = image.load_img('2.jpg', target_size=(64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.array(test_image, dtype=np.float32)
# test_image = np.reshape(test_image, (-1, 64, 64, 1))

# result = model.predict(test_image)[0]

# print("result")
# print(result)

# pred_class = list(result).index(max(result))
# print("pred class")
# print(pred_class, dirs[pred_class])


# test_image = cv2.imread('2.jpg', cv2.COLOR_BGR2GRAY)
# resizedImg = cv2.resize(test_image, (64, 64))
# test_image = cv2.GaussianBlur(resizedImg,(5,5),2)



image = cv2.imread('Z.jpg', cv2.COLOR_BGR2GRAY)
image = np.array(image, dtype=np.float32)
# test_image = cv2.GaussianBlur(resizedImg,(5,5),2)
image = np.reshape(image, (-1, 64, 64, 1))


result = model.predict(image)

print("result")
print(result)

# pred_class = list(result).index(max(result))
# print("pred class")
# print(pred_class, dirs[pred_class])

