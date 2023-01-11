import cv2
import numpy as np
import glob
import os

#dirs = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
dirs = ["1", "2", "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

for j in range(2):
    mode = "train/"
    if(j == 1):
        mode = "test/"
    
    for i in range(len(dirs)):
        parent_dir = "/home/sujoy/college-project-repo/Sign-Language-to-Text-master/preprocessed-data/" + mode
        path = os.path.join(parent_dir, dirs[i])

        #creating directory to store the preprocessed image of each letter
        try:
            os.makedirs(path)
            print("created dir /preprocessed-data/" + mode + dirs[i])
        except OSError as error:
            print(error)

        #reading all files of that letter
        images = [cv2.imread(file,0) for file in glob.glob("data/" + mode + dirs[i] + "/*.jpg")]

        print(str(len(images)) + " images read of sign " + dirs[i])

        imageName = 0

        #applying filters and storing image
        for img in images:
            resizedImg = cv2.resize(img, (30, 40))
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(resizedImg,(5,5),2)

            opPath = "preprocessed-data/" + mode + dirs[i] + "/" + str(imageName) + ".jpg"
            cv2.imwrite(opPath, blur)
            imageName += 1