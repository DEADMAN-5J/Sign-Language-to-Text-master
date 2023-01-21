import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Convolution2D, Dense, MaxPool2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

train_dir = "preprocessed-data/train"
test_dir = "preprocessed-data/test"
img_width = 64
img_height = 64
batch_size = 20

datagen = ImageDataGenerator(rescale=1.0/255)

train_data_gen = datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), class_mode='categorical', color_mode='grayscale')

test_data_gen = datagen.flow_from_directory(test_dir, target_size=(img_width, img_height), class_mode='categorical', color_mode='grayscale')

train_data_gen.class_indices
train_data_gen.classes

model = Sequential()
# model.add(Convolution2D(64, kernel_size=3 , input_shape=(img_width, img_height, 1), activation='relu'))
# model.add(MaxPool2D(3,3))

# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(35, activation='softmax'))



#soumick's layers
# First convolution layer and pooling
model.add(Convolution2D(32, (3, 3), input_shape=(img_height, img_width, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
# Second convolution layer and pooling
model.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
#model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
model.add(Flatten())

# Adding a fully connected layer
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.40))
model.add(Dense(units=96, activation='relu'))
model.add(Dropout(0.40))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=35, activation='softmax')) # softmax for more than 2
#soumick's layers



model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data_gen, steps_per_epoch=len(train_data_gen), epochs=5, validation_data=test_data_gen, validation_steps=len(test_data_gen))


# Saving the model
model_json = model.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
model.save_weights('model-bw.h5')
print('Weights saved')

print("history")
print(history.history)


#plotting accuracy and training loss
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="Validation accuracy")
plt.title("Validation accuracies")
plt.ylabel("accuracy")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.show()

plt.plot(history.history['loss'], label="Training loss")
plt.plot(history.history['val_loss'], label="Validation loss")
plt.title('Validation loss values')
plt.ylabel("loss value")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.show()
