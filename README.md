# leaf_health

Data obtained from: https://data.mendeley.com/datasets/hb74ynkjcn/5
All the images are being ignored for GitHub upload due to their size being too large.
However, they are being saved and used in local directory.

Objective:
The objective of this project is to create a model and setup that is able to intake a plant's leaf image and categorize its species and health via comparison to publicly available dataset.

Approach:
I plan to take a multi-step approach for my project. First step was to find an appropriate dataset that would have images of plants of some sort and include categorization of healthy vs unhealthy. This step is complete (See References). Second step is to read and review the citations of this data source to better understand the dataset and its usage. Third step is to create a setup that can access and read this dataset (images), preferably via Python. Fourth step is to build a sample model that is able to categorize the images - this will be done via a sample set of images separated from the training dataset. Fifth step is to optimize and improve the model. Lastly, sixth step is to implement and test the model on images taken from my garden.

Image Segmentation support resources;
https://cnvrg.io/image-segmentation/

Data split resources;
https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn

Image Feature Extraction resources (and SVM model building);
https://rpubs.com/Sharon_1684/454441


Steps to recreate this project;
1. Image acquisition
2. Image pre-processing via img_preprocess.py
3. Model build and test via main.py
