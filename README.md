# Image-Classification-with-SVM

SVM IMAGE CLASSIFICATION

Author: acelik8

Dataset link: https://www.kaggle.com/c/dogs-vs-cats

Input: either sixtyFour.pickle or the thirtyTwopickle.pickle files 
       sixtyFour.pickle is the original dataset converted to 64x64 resolution
       thirtyTwo.pickle is the original dataset converted to 32x32 resolution
       The code for converting the image input to different resolutions is provided.
       In order to change the size of the dataset of preference(32 or 64 resolution),      
       variable "sayi" in the code can be changed. The dataset size will be equal to 
       the two times of the value of "sayi" specified.

Output: Different model files of different dataset sizes in the models_32x32Size folder.
	Model files from 1 to 5 representing models trained on datasets of following 		sizes in order: 2000,4000,6000,8000,10000


execution: python3 SVM_image_classification.py
