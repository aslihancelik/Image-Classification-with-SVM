""" Author: acelik """

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pickle
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

dir = '/Users/aslihancelik/Desktop/hw1/kagglecatsanddogs_3367a/Petimages'

class_categories = ['Cat', 'Dog']

data = []

#random.seed(3)


"""testing out the images in dataset initially"""
#for category in class_categories:
#	path = os.path.join(dir,category)
#	print(path)
#
##     for img in os.listdir(path):
##         imgpath = os.path.join(path,img)
##         pet_img = cv2.imread(imgpath)
##         cv2.imshow('image', pet_img)
##         break
##     break
#
## cv2.waitKey(0)
## cv2.destroyAllWindows()

""" Getting the image data"""
#
#for category in class_categories:
#    path = os.path.join(dir,category)
#    label = class_categories.index(category)  #0 for cat and 1 for dog
#
#    for img in os.listdir(path):
#        imgpath = os.path.join(path,img)
#        pet_img = cv2.imread(imgpath,1)
#
#        try:
#            pet_img = cv2.resize(pet_img,(64,64))
#            image = np.array(pet_img).flatten()
#
#            data.append([image,label])
#        except Exception as e:
#            pass
#



"""to save the image data into pickle file"""
#pick_in = open('/Users/aslihancelik/Desktop/hw1/sixtyFour.pickle', 'wb')
#pickle.dump(data,pick_in)
#pick_in.close()

"""getting the image data from data.pickle file"""
pick_in = open('/Users/aslihancelik/Desktop/CS6220_HW1_acelik8/sixtyFour.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()



random.shuffle(data)
features = []
labels = []


for feature,label in data:
    features.append(feature)
    labels.append(label)
    


"""to check if there is equal amounts of pictures from two classes"""
#catt=0
#dogg=0
#for i in range (len(labels)):
#    if labels[i]==0 :
#        catt+=1
#    else:
#        dogg+=1
#
#print(catt , dogg)
    

"""to get equal amounts of pictures from each classes"""
y=0
for i in range (len(labels)):
    if labels[i]==0 :
        del labels[i]
        del features[i]
        y+=1
    if y==6:
        break
            
#print(len(labels),len(features))

"""to check if there is equal amounts of pictures from two classes"""
yayy=0
mayy=0
for i in range (len(labels)):
    if labels[i]==0 :
        yayy+=1
    else:
        mayy+=1

#print(yayy , mayy)

"""getting the first 10 images from shuffled data"""
features_10 = []
labels_10 = []

say=0
say1=0
sayi=1000      #2 times of this number determines the size of the dataset
for i in range (len(labels)):
        if labels[i]==0 and say<sayi:
            features_10.append(features[i])
            labels_10.append(labels[i])
            say +=1
        elif labels[i]==1 and say1<sayi:
            features_10.append(features[i])
            labels_10.append(labels[i])
            say1 +=1

#print(say,say1)
#print(len(features_10),len(labels_10))



xtrain , xtest , ytrain, ytest = train_test_split(features_10, labels_10, test_size=0.2)
#xtrain , xtest , ytrain, ytest = train_test_split(features, labels, test_size=0.2)

#print(len(xtrain),len(xtest),len(ytrain),len(ytest))

"""Parameter Experiments"""
#model = SVC(C=1,kernel='linear', gamma= 'auto',verbose=2)
#model = SVC(C=0.1,kernel='poly', degree=3, gamma= 1,verbose=2, max_iter=4000)
#model = SVC(C=1,kernel='poly', gamma= 'auto',verbose=2)
#model = SVC(C=1,kernel='poly', gamma= 'auto',verbose=2, max_iter = 50)
#model=SVC(verbose=2)
#print(model)
"""Optimal Model according to GridSearch"""
model = SVC(C=0.1,kernel='poly', degree=3, gamma= 1,verbose=2)
model.fit(xtrain, ytrain)


"""saving the model"""
#pick = open('model_sixtyFour.sav', 'wb')
#pickle.dump(model,pick)
#pick.close()

"""getting the size of the model"""
#model_size = os.path.getsize('model_sixtyFour.sav')
#print(model_size)

"""loading the model"""
#pick = open('model_sixtyFour_next2.sav', 'rb')
#model= pickle.load(pick)
#pick.close()


'''Tuning the SVM parameters by evaluating Kernels'''
#kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']
##A function which returns the corresponding SVC model
#def getClassifier(ktype):
#    if ktype == 0:
#        # Polynomial kernel
#        return SVC(kernel='poly',gamma="auto")
#    elif ktype == 1:
#        # Radial Basis Function kernal
#        return SVC(kernel='rbf', gamma="auto")
#    elif ktype == 2:
#        # Sigmoid kernal
#        return SVC(kernel='sigmoid', gamma="auto")
#    elif ktype == 3:
#        # Linear kernal
#        return SVC(kernel='linear', gamma="auto")
#
#for i in range(4):
#    # Separate data into test and training sets
##    xtrain , xtest , ytrain, ytest = train_test_split(features_10, labels_10, test_size=0.2)
#    #model using different kernal
#    svclassifier = getClassifier(i)
#    svclassifier.fit(xtrain, ytrain)# Make prediction
#    y_pred = svclassifier.predict(xtest)# Evaluate our model
#    print("Evaluation:", kernels[i], "kernel")
#    print(classification_report(ytest,y_pred))

'''Parameter Tuning for GridSearch'''
#param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1,0.1,0.01,0.001,'auto'], 'kernel': ['linear', 'poly', 'sigmoid', 'rbf']}
#
#grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
#grid.fit(xtrain,ytrain)
#
#print(grid.best_estimator_)
#
#grid_predictions = grid.predict(xtest)
#
#print(classification_report(ytest,grid_predictions))


#print(confusion_matrix(ytest,grid_predictions))

'''GridSearch Result Model'''
#model = SVC(C=0.1,kernel='poly', degree=3, gamma= 1,verbose=2)
#model.fit(xtrain, ytrain)



'''checking the change in accuracy according to tolerance parameter''' #no significant pattern, not in the report
#toler=[]
#acc = []
#tol_default = 0.001
#for i in range(400):
#    toler.append(tol_default)
#    # Separate data into test and training sets
##    xtrain , xtest , ytrain, ytest = train_test_split(features_10, labels_10, test_size=0.2)
#    #model using different kernal
#    svclassifier = SVC(C=0.1,kernel='poly', degree=3, gamma= 1, tol=tol_default)
#    svclassifier.fit(xtrain, ytrain)# Make prediction
##    y_pred = svclassifier.predict(xtest)# Evaluate our model
##    print("tol value:", tol_default)
#    acc_svm = svclassifier.score(xtest,ytest)
#    acc.append(acc_svm)
##    print('Accuracy:',acc_svm))
#    tol_default+=0.01
#
#plt.plot(toler,acc)
#plt.show()
    

'''Accuracy of the Model and Prediction'''

prediction = model.predict(xtest)

accuracy = model.score(xtest,ytest)


print(classification_report(ytest,prediction))

print('Accuracy:', accuracy)

print('Prediction is :',class_categories[prediction[0]] )

mypet=xtest[0].reshape(64,64,-1)

plt.imshow(mypet)
plt.show()


'''Plotting Model Accuracy vs Model Size'''
#model=['/content/model_thirtyTwo_first.sav', '/content/model_thirtyTwo_second.sav', '/content/model_thirtyTwo_third.sav', '/content/model_thirtyTwo_fourth.sav', '/content/model_thirtyTwo_fifth.sav']
#acc = []
#
#xtrain , xtest , ytrain, ytest = train_test_split(features_10, labels_10, test_size=0.2)
#
#
#pick = open('/content/model_thirtyTwo_first.sav', 'rb')
#model= pickle.load(pick)
#pick.close()
#
#accuracy = model.score(xtest,ytest)
#acc.append(accuracy)
#
#pick = open('/content/model_thirtyTwo_second.sav', 'rb')
#model2= pickle.load(pick)
#pick.close()
#
#accuracy = model2.score(xtest,ytest)
#acc.append(accuracy)
#
#pick = open('/content/model_thirtyTwo_third.sav', 'rb')
#model3= pickle.load(pick)
#pick.close()
#
#accuracy = model3.score(xtest,ytest)
#acc.append(accuracy)
#
#pick = open('/content/model_thirtyTwo_fourth.sav', 'rb')
#model4= pickle.load(pick)
#pick.close()
#
#accuracy = model4.score(xtest,ytest)
#acc.append(accuracy)
#
#pick = open('/content/model_thirtyTwo_fifth.sav', 'rb')
#model5= pickle.load(pick)
#pick.close()
#
#accuracy = model5.score(xtest,ytest)
#acc.append(accuracy)
#
#
#x=[2000,4000,6000,8000,10000]
## x=[1000,2000]
#plt.tight_layout()
#plt.plot(x,acc)
#plt.title('Dataset Size vs Model Accuracy')
#plt.xlabel('Binary Classifiers')
#plt.ylabel('Accuracy')
#plt.xticks(np.arange(min(x), max(x)+1, 2000))
#plt.show()


""" Dataset from: https://www.kaggle.com/c/dogs-vs-cats """
""" Reference link:  https://www.youtube.com/watch?v=0rjlviOQlbc """
""" for iteration tracking: https://stackoverflow.com/questions/41486610/knowing-the-number-of-iterations-needed-for-convergence-in-svr-scikit-learn """
""" Kernels fucntion: https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/ """
""" https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html """

