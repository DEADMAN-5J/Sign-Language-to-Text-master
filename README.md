# Sign-Language-to-Text

Project converts American sign language to text in realtime. It uses CNN to train the required models for prediction. The dataset is custom made.

dataset :

- train : https://drive.google.com/drive/u/1/folders/1-XTAjPPRPFeRqu3848z8dMXaolILWizn
- test : https://drive.google.com/drive/u/1/folders/18e1F1n1SWPF8lUF8pCKdUzSzKAbmSbVN

Demo : https://www.youtube.com/watch?v=aU5-8XJrxwY&t=2s

# How the project works

## #1 collecting data

To create your own dataset, the collect-data.py file is used. To use this, we need to run the file, show a letter's sign in the filtered frame(can use signs images as reference), then press the letter being shown in the keyboard. This will start collecting training image data and store in the /data/train folder. Once desired ammount of train image is generated, we can press esc to exit form app, and start collecting test data by repreating the same procedure and changing the mode to "test" by changing line 27.

## #2 data preprocessing

To preprocess the images, first we will scale down the image from 300x400 to 64x64, then apply two filters one is black&white then gaussian. The result images will be stored in preprocessed-data folder. To do all these, we need to set the root directory of the project by changing 15th line of image_preprocessing.py, then run the image_preprocessing.py file.

testing-filters.py is used to show how the data images will look after perprocessing.

## #3 training model and showing accuracy

After data preprocessing, then comes the model training part. For that we need to run train2.py file. The job of this file is to train the model, store it locally, and plot the training accuracy and loss in charts. after training the model, it generates model-bw.h5 and model-bw.json files, they are then usually stored in the model directory.

## #4 using the trained model to predict hand signs

once we have the training models, we can use the app2.py file to get sign inputs from device camera and predict the sign.
