import csv
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import cv2 
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if (not os.environ.get('PYTHONHTTPSVERIFY','')and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('Pic.npz')['arr_0']
y = pd.read_csv('Project122.csv')['labels']
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
no_of_classes = len(classes)


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
x_train = x_train/255
x_test = x_test/255

lr = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train.value,y_train)

y_prediction = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)

imgCapture = cv2.VideoCapture(0)
while(True):
    try:
        ret ,frame = imgCapture.read()
        imgColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
        height = imgColor.shape[0]
        width = imgColor.shape[1]
        upper_left = (int(width/2-56), int(height/2-56))
        bottom_right = (int(width/2+56), int(height/2+56))
        cv2.rectangle(imgColor, upper_left, bottom_right, (0,255,0), 2)
        roi = imgColor[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        imgPIL = Image.fromarray(roi)
        imgBW = imgPIL.convert('l')
        imgBW_resized = imgBW.resize((28,28), Image.LANCZOS)
        imgBW_ri = PIL.ImageOps.invert(imgBW_resized)
        pixel_filter= 20
        minpixel = np.percentile(imgBW_ri, pixel_filter)
        imgBW_ris = np.clip(imgBW_ri-minpixel,0,255 )
        maxpixel = np.max(imgBW_ri)
        imgBW_ris = np.asarray(imgBW_ris)/maxpixel

        test_sample = np.array(imgBW_ris).reshape(1,784)
        test_predict = lr.predict(test_sample)
        print("Predicted class is ", test_predict)
        cv2.imshow("Frame", imgColor)

        if cv2.waitKey(1) & 0xFF== ord('q'):
            break

    except Exception as e:
        pass

imgCapture.release()
cv2.destroyAllWindows()

