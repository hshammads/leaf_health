import os
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import numpy as np
import PIL
import PIL.Image as pilim
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import cv2
import glob
import pickle
from tensorflow import keras
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

# define constants
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
PROJECT_DIR = os.getcwd()
INPUT_DIR = PROJECT_DIR + 'Preprocessed Data/'
RESIZED_INPUT_DIR = PROJECT_DIR + 'Preprocessed Data_Resized/'
RESIZED_AND_CATEGORIZED_INPUT_DIR = PROJECT_DIR + 'Preprocessed Data - Resized and Categorized/'

# categories dictionary
CATEGORIES_DICTIONARY = {'0001': 'Mango - Healthy', '0002': 'Arjun - Healthy',
                         '0003': 'Alstonia Scholaris - Healthy', '0004': 'Guava - Healthy',
                         '0005': 'Jamun - Healthy', '0006': 'Jatropha - Healthy',
                         '0007': 'Pongamia Pinnata - Healthy', '0008': 'Basil - Healthy',
                         '0009': 'Pomegranate - Healthy', '0010': 'Lemon - Healthy',
                         '0011': 'Chinar - Healthy', '0012': 'Mango - Diseased',
                         '0013': 'Arjun - Diseased', '0014': 'Alstonia Scholaris - Diseased',
                         '0015': 'Guava - Diseased', '0016': 'Bael - Diseased',
                         '0017': 'Jamun - Diseased', '0018': 'Jatropha - Diseased',
                         '0019': 'Pongamia Pinnata - Diseased', '0020': 'Pomegranate - Diseased',
                         '0021': 'Lemon - Diseased', '0022': 'Chinar - Diseased'}

# define global variables
features_list = []
target_array = []

# define functions
# function to create image features and flatten into a single row
def create_features(image):
    # flatten three channel color image
    color_features = image.flatten()
    # get HOG features from image
    hog_features, hog_image = hog(image, visualize = True, block_norm = 'L2-Hys', pixels_per_cell = (16,16), channel_axis = -1)
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    return flat_features

# ask user for debug mode
debug = input('\nWould you like the debug mode on? (Y/N): ')
if debug.upper() == 'Y' or debug.upper() == 'YES':
    debug = True
else:
    debug = False

if debug:
    print('Debug mode is ON ...')
else:
    print('Debug mode is OFF ...')

# ask user for SVM model type
# kernal types: linear kernel = 'linear', polynomial kernel = 'poly', radial basis kernel ('rbf'), sigmoid kernel ('sigmoid').
model_kernel_type = input('\nWhich SVM model type would you like to run? (linear/l, poly/p, rbf/r, sigmoid/s): ')
if model_kernel_type.upper() == 'S' or model_kernel_type.upper() == 'SIGMOID':
    model_kernel_type = 'sigmoid'
    if debug:
        print(model_kernel_type, 'SVM model selected.')
elif model_kernel_type.upper() == 'R' or model_kernel_type.upper() == 'RBF':
    model_kernel_type = 'rbf'
    if debug:
        print(model_kernel_type, 'SVM model selected.')
elif model_kernel_type.upper() == 'P' or model_kernel_type.upper() == 'POLY':
    model_kernel_type = 'poly'
    if debug:
        print(model_kernel_type, 'SVM model selected.')
elif model_kernel_type.upper() == 'L' or model_kernel_type.upper() == 'LINEAR':
    model_kernel_type = 'linear'
    if debug:
        print(model_kernel_type, 'SVM model selected.')
else:
    model_kernel_type = 'linear'
    if debug:
        print('Invalid input! Program will run with', model_kernel_type, 'SVM model.')

# check if the stated model type exists already
# if it does, ask whether it should be rebuilt (default answer should be no)
model_name = 'img_hog_' + model_kernel_type + '_model.p'
model_path = PROJECT_DIR + model_name

rebuild_model = True
model_exists = os.path.isfile(model_path)

if model_exists:
    rebuild_model = input('{} {}'.format(model_kernel_type, 'SVM model already exists!\nWould you like to rebuild the model (NOT RECOMMENDED)? (Y/N): '))
    if rebuild_model.upper() == 'Y' or rebuild_model.upper() == 'YES':
        rebuild_model = True
        if debug:
            print(model_kernel_type, 'SVM model will be rebuilt. This will take a while ...')
    else:
        rebuild_model = False
        if debug:
            print(model_kernel_type, 'SVM model will not be rebuilt!')

# model will only be built if it doesn't already exist or the user wants it to be rebuilt
if rebuild_model:
    # pull in images, run HOG, create image features and flatten into a single row
    print('Starting image import and feature creation ...')
    # set start timer
    start = timer()

    image_index = 0
    for filename in glob.iglob(RESIZED_INPUT_DIR + '**/*.JPG', recursive = True):
        # read image
        image = cv2.imread(filename)
        image_index += 1

        # create target array
        image_name = Path(filename).name
        image_name_category = CATEGORIES_DICTIONARY.get(image_name[:4])
        target_array.append(image_name_category)

        # create image features
        image_features = create_features(image)
        if debug:
            print('Image index:', image_index, '; Image Name:', image_name, '; Image Features:', image_features, '; Image Category:', image_name_category)

        # create feature matrix
        features_list.append(image_features)

    print('Image import and feature creation completed successfully!')

    # create feature matrix
    print('Creating feature matrix ...')
    feature_matrix = np.array(features_list)
    if debug:
        print('Feature matrix:', feature_matrix)
    print('Feature matrix created successfully!')

    # get shape of feature matrix
    if debug:
        print('Feature matrix shape is: ', feature_matrix.shape)

    # Scale feature matrix + PCA
    # define standard scaler
    print('Scaling feature matrix and running PCA ...')
    ss = StandardScaler()
    # run this on our feature matrix
    stand = ss.fit_transform(feature_matrix)

    pca = PCA(n_components = 500)
    # use fit_transform to run PCA on our standardized matrix
    leaf_pca = ss.fit_transform(stand)
    # look at new shape
    if debug:
        print('PCA matrix shape is: ', leaf_pca.shape)

    # data split
    print('Starting data split ...')
    X = pd.DataFrame(leaf_pca) # dataframe
    y = pd.Series(np.array(target_array))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 1234123) # original data split

    # below is for splitting data into train, validate, test
    testing_ratio = 0.15
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testing_ratio, random_state = 1234123, stratify = y) # testing data is 15%
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size = testing_ratio / (1 - testing_ratio), random_state = 1234123, stratify = y) # validation data is 15% [x% of 85%]

    print('Data split successfully!')

    # look at the distrubution of categories in the train set
    if debug:
        print('Categories distribution:\n',pd.Series(y_train).value_counts())

    # train model
    print('Training model using ' + model_kernel_type + ' SVM ...')
    # define support vector classifier
    svm = SVC(kernel = model_kernel_type, probability = True, random_state = 42)

    # fit model
    svm.fit(X_train, y_train)
    print('The model has been trained well with the given images!')

    # score model
    print('Testing model accuracy ...')
    # generate predictions
    y_pred = svm.predict(X_val)
    print('The predicted data is:', y_pred)
    print('The actual data is:', np.array(y_val))
    print('The model (', model_kernel_type, 'SVM) is', str(round(accuracy_score(y_pred, y_val)*100, 2)), '% accurate')

    # ROC Curve + AUC
    # predict probabilities for X_test using predict_proba
    probabilities = svm.predict_proba(X_val)

    # select the probabilities for label 1.0
    y_proba = probabilities[:, 1]

    # calculate false positive rate and true positive rate at different thresholds
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_proba, pos_label = 1)

    # calculate AUC
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    # plot the false positive rate on the x axis and the true positive rate on the y axis
    roc_plot = plt.plot(false_positive_rate, true_positive_rate, label = 'AUC = {:0.2f}'.format(roc_auc))

    # plt.legend(loc = 0)
    # plt.plot([0,1], [0,1], ls = '--')
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate');


    #--------------------------------------------------------------------#
    # save model to disk
    if debug:
        print('Saving model to disk ...')

    filehandler = open(model_name,'wb')
    pickle.dump(svm,filehandler)
    filehandler.close()

    if debug:
        print('Pickle is dumped successfully!')

    # set end timer
    end = timer()
    run_time = int(end - start)
    if run_time < 60:
        time_taken = str(run_time) + ' second(s)'
    elif run_time >= 60 and run_time < 3600:
        time_taken = str(int(run_time / 60)) + ' minute(s) and ' + str(run_time % 60) + ' second(s)'
    else:
        time_taken = str(int(run_time / 3600)) + ' hour(s) and ' + str(int((run_time % 3600) / 60)) + ' minute(s) and ' + str((run_time % 3600) % 60) + ' second(s)'

    print('Process completed in: ' + time_taken)

#--------------------------------------------------------------------#
# model evaluation
else:
    # load model
    filehandler = open(model_name,'rb')
    model = pickle.load(filehandler)
    if debug:
        print('Pickle is loaded successfully!')
    filehandler.close()
