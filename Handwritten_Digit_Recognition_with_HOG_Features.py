from keras.datasets import mnist
from skimage.feature import hog
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import svm


#================================ STEP 1: Load Dataset =================================================================
(train_Img, train_Labels), (test_Img, test_Labels) = mnist.load_data()

# Set-up training and testing images and labels
#Training
Train_Im_0 = train_Img[train_Labels == 0,:,:]
Train_Lb_0 = train_Labels[train_Labels == 0]
Train_Im_1 = train_Img[train_Labels == 1,:,:]
Train_Lb_1 = train_Labels[train_Labels == 1]
Train_Im_2 = train_Img[train_Labels == 2,:,:]
Train_Lb_2 = train_Labels[train_Labels == 2]
Train_Im_3 = train_Img[train_Labels == 3,:,:]
Train_Lb_3 = train_Labels[train_Labels == 3]
Train_Im_4 = train_Img[train_Labels == 4,:,:]
Train_Lb_4 = train_Labels[train_Labels == 4]
Train_Im_5 = train_Img[train_Labels == 5,:,:]
Train_Lb_5 = train_Labels[train_Labels == 5]
Train_Im_6 = train_Img[train_Labels == 6,:,:]
Train_Lb_6 = train_Labels[train_Labels == 6]
Train_Im_7 = train_Img[train_Labels == 7,:,:]
Train_Lb_7 = train_Labels[train_Labels == 7]
Train_Im_8 = train_Img[train_Labels == 8,:,:]
Train_Lb_8 = train_Labels[train_Labels == 8]
Train_Im_9 = train_Img[train_Labels == 9,:,:]
Train_Lb_9 = train_Labels[train_Labels == 9]

#Testing
Test_Im_0 = test_Img[test_Labels == 0,:,:]
Test_Lb_0 = test_Labels[test_Labels == 0]
Test_Im_1 = test_Img[test_Labels == 1,:,:]
Test_Lb_1 = test_Labels[test_Labels == 1]
Test_Im_2 = test_Img[test_Labels == 2,:,:]
Test_Lb_2 = test_Labels[test_Labels == 2]
Test_Im_3 = test_Img[test_Labels == 3,:,:]
Test_Lb_3 = test_Labels[test_Labels == 3]
Test_Im_4 = test_Img[test_Labels == 4,:,:]
Test_Lb_4 = test_Labels[test_Labels == 4]
Test_Im_5 = test_Img[test_Labels == 5,:,:]
Test_Lb_5 = test_Labels[test_Labels == 5]
Test_Im_6 = test_Img[test_Labels == 6,:,:]
Test_Lb_6 = test_Labels[test_Labels == 6]
Test_Im_7 = test_Img[test_Labels == 7,:,:]
Test_Lb_7 = test_Labels[test_Labels == 7]
Test_Im_8 = test_Img[test_Labels == 8,:,:]
Test_Lb_8 = test_Labels[test_Labels == 8]
Test_Im_9 = test_Img[test_Labels == 9,:,:]
Test_Lb_9 = test_Labels[test_Labels == 9]
#=======================================================================================================================



#================================ STEP 2: HOG Features Calculation======================================================

HOG_Test = []
HOG_Train = []
LABEL_TEST = []
LABEL_TRAIN = []



for j in range(0,10,1):
    print('Digit: ', j)
    if j == 0:
        Test_In = Test_Im_0
        Train_In = Train_Im_0
        n1 = np.size(Test_Im_0, 0)
        n2 = np.size(Train_Im_0, 0)
    elif j == 1:
        Test_In = Test_Im_1
        Train_In = Train_Im_1
        n1 = np.size(Test_Im_1, 0)
        n2 = np.size(Train_Im_1, 0)
    elif j == 2:
        Test_In = Test_Im_2
        Train_In = Train_Im_2
        n1 = np.size(Test_Im_2, 0)
        n2 = np.size(Train_Im_2, 0)
    elif j == 3:
        Test_In = Test_Im_3
        Train_In = Train_Im_3
        n1 = np.size(Test_Im_3, 0)
        n2 = np.size(Train_Im_3, 0)
    elif j == 4:
        Test_In = Test_Im_4
        Train_In = Train_Im_4
        n1 = np.size(Test_Im_4, 0)
        n2 = np.size(Train_Im_4, 0)
    elif j == 5:
        Test_In = Test_Im_5
        Train_In = Train_Im_5
        n1 = np.size(Test_Im_5, 0)
        n2 = np.size(Train_Im_5, 0)
    elif j == 6:
        Test_In = Test_Im_6
        Train_In = Train_Im_6
        n1 = np.size(Test_Im_6, 0)
        n2 = np.size(Train_Im_6, 0)
    elif j == 7:
        Test_In = Test_Im_7
        Train_In = Train_Im_7
        n1 = np.size(Test_Im_7, 0)
        n2 = np.size(Train_Im_7, 0)
    elif j == 8:
        Test_In = Test_Im_8
        Train_In = Train_Im_8
        n1 = np.size(Test_Im_8, 0)
        n2 = np.size(Train_Im_8, 0)
    elif j == 9:
        Test_In = Test_Im_9
        Train_In = Train_Im_9
        n1 = np.size(Test_Im_9, 0)
        n2 = np.size(Train_Im_9, 0)

    for i in range(0, n1, 1):
        fd1, hog_image = hog(Test_In[i, :, :], orientations=9, pixels_per_cell=(6, 6),
                             cells_per_block=(2, 2), visualize=True, multichannel=False)
        HOG_Test.append(fd1)
        LABEL_TEST.append(j)


    for k in range(0, n2, 1):
         fd2, hog_image = hog(Train_In[k, :, :], orientations=9, pixels_per_cell=(6, 6),
                             cells_per_block=(2, 2), visualize=True, multichannel=False)
         HOG_Train.append(fd2)
         LABEL_TRAIN.append(j)

#=======================================================================================================================




#================================ STEP 3: Set up and run SVM============================================================
# Run SMV
SVMModel = svm.SVC(kernel='linear', decision_function_shape='ovo').fit(HOG_Train, LABEL_TRAIN)

# Predict Test Labels
Predicted_Digits = SVMModel.predict(HOG_Test)

# Create and print Confusion Matrix
cm_lin = confusion_matrix(LABEL_TEST, Predicted_Digits)
print(cm_lin)
#=======================================================================================================================


