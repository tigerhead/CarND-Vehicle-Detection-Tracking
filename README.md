

## Vehicle Detection Project

This is project 5 in Udacity Self-driving car nano-degree project term 1. A software pipeline is implemented to identify vehicles in a video from a front-facing camera on a car in this project. The goals / steps of this project are the following:
    
     1. Apply a color transform and append binned color features, as well as histograms of color
    
    2. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training 
       set of images and train a classifier Linear SVM classifier, append HOG vector to 
       color feature vector   
    
    3. Extract Features from training images to train selected classier
    
    4. Implement a sliding-window technique and use trained classifier to search for vehicles in images.
    
    5. Use heatmap technique to eliminate negative positive detection.
    
    6. Use label to find continuous non-zero area in threshold heat map and draw single 
       box for detected vehicles
    
    7. Run the pipeline created from step 4-6 on a video stream  and 
       draw outlined box on vehicle in the video

List of files and folders:

P5.ipynb  -  The main jupyter notebook file which has step by step description and code spinet for this project

P5-Test-Classifier-HSV.ipynb, 
P5-Test-Classifier-RGB.ipynb, 
P5-Test-Classifier-HSL.ipynb, 
P5-Test-Classifier-LUV.ipynb, 
P5-Test-Classifier-YCrCb.ipynb -- jupyter notebook files to test features extracted with different color sapce and differet classifier.

P5_solution_v5.mp4 - Solution video

test_images folder  - Contains all test images

output_images folder - Contains all output image in the project to demo the pipeline


### Feature Extraction.  

I choose to use 3 type of features in this projecct: 

1) spatial binning of color

2) color histogram  

3) HOG (Historgram of Oriented Gradient).

#### Spatial Binning of Color

Extract feature from raw pixel values of an image. The image from training data is resized into given size like (32, 32) and converted into a vector using openCV ravel(). Code can be found in code cell No. 3 in P5.ipynb. 

Spatial size used in the project:  (16, 16) # Spatial binning dimensions


![Camera Calibration](/output_images/spa_bin_feat_ex.png)

#### Histograms of Color

Color is an important feature for vihecle detection,  so color histgram features are etracted. Code can be found in code cell No. 4 in P5.ipynb. Number of histogram bins used in this project is 32.

#### Histogram of Oriented Gradients (HOG)
I choose to use dataset provided by Udactiy for classifier training and testing. All the `vehicle` and `non-vehicle` images are loaded to a dictionary. Code can be found in code cell No. 5 in P5.ipynb. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car-noncar](/output_images/car_noncar_ex.png)

A car can be distinguished from a non-car by looking at its edges. HOG will compute the gradients from blocks of cells. Then, a histogram is constructed with these gradient values. skimage.feature.hog function is used to extract HOG feature from image. I then explored different color spaces like RGB, LUV, HSV, HLS, YCrCb and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  Code can be found in code cell No. 6 in P5.ipynb. The comparison between different color space and classifier can be found in P5-Test-Classifier-HSV.ipynb, P5-Test-Classifier-RGB.ipynb, P5-Test-Classifier-HSL.ipynb, P5-Test-Classifier-LUV.ipynb, P5-Test-Classifier-YCrCb.ipynb. To balance accuracy of classifier and training time, I choose to use follwoing HOG feature extration parameters:

color_space = 'YCrCb' 

orient = 9  # HOG orientations

pix_per_cell = 8 # HOG pixels per cell

cell_per_block = 2 # HOG cells per block

hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"


Here are examples of car and non car HOG feature image:

![car-HOG](/output_images/car_hog_ex.png)

![noncar-HOG](/output_images/noncar_hog_ex.png)



### Train Classifier.

LinearSVC, SVM, RandomForestClassifier, DecisionTreeClassifier tree and AdaBoostClassifier  provided by sklearn are tested with different color spaces: like RGB, LUV, HSV, HLS, YCrCb. The result can be found in different notebook. For example: RGB with different classifier, see result in P5-Test-Classifier-RGB.ipynb. It looks like YCrCb space with LinearSVC classifier gave best result. So in final project report, YCrCb color space and LinearSVC classifier is used. 

Features are extracted for each iamge in provided dataset. Feature vectors are normalized. Dataset are randomly slit intot train and test dataset. Code can be found in code cell No. 9, 10 and 11 in P5.ipynb. The accuray for the classifier is 0.9845


### Sliding Window Technique

Slide widow technique is used to search vehicle on images. The image to be processed is divided small windows. Code can be found in code cell No. 13 in P5.ipynb. Here is an example which devided image into (64, 64) size windows


![sliding_Windows](/output_images/sliding_win_ex.png)



Trained classifier is uesd to classify each window and windows classified as vehicle will be saved to be used to mark the position of vehicle in the image. Code can be found in code cell No. 12 in P5.ipynb. Here is an example of one test image marked with windows classified as car, window size is (64, 64)


![single_Windows](/output_images/car_sin_win_ex.png)

---

### Video Implementation

#### Multiple sliding windows

Multiple sildeing window sizes are used to search vehicles in the image to ensure vehicle are not missed. Code can be found in code cell No. 14 in P5.ipynb. I tried different combination of window sizes and overlap rate, finally chose follow three sizes: (64, 64), (96, 96), (128, 128) and overlap rate 0.75. Here is another example:

![multi_Windows](/output_images/car_multi_win_ex.png)


#### Reducing false positive and combine multiple boxes.

All positve boxes are clollected as demostrated in example above.   From the positive detections,  I created a heatmap and then thresholded that map to identify true vehicle positions and eliminate false positive. Code can be found in code cell No. 16 in P5.ipynb. Here is example of heat map and threshold heat map. Th false positive which is shade in a tree is elimimated.

![heat_map](/output_images/heatmap_ex.png)


![threshold_heat_map](/output_images/heatmap_thr_ex.png)


label from scipy.ndimage.measurements is used to find contineous nonzero areas and label them starting from 1 and set the background as 0. And min and max point will be picked will be used to draw single box around detected vehicle. Code can be found in code cell No. 18 in P5.ipynb. Here is an example:

![label_heat_map](/output_images/label_heatmap_ex.png)


![final_result](/output_images/result_ex.png)


#### Assemble pipeline
Following steps are assembled together to implement the pipeline to process single image frame in the video:

    1) Use liding-window technique to identify postive windows contains vehicle object.     
    
    2) Use heatmap technique to eliminate negative positive detection.
    
    3) Use label to find continuous non-zero area in threshold heat map and draw single box for detected vehicles
    
 Code can be found in code cell No. 20 in P5.ipynb

 Here are test images processed by the pipeline:
 

![final_result_1](/output_images/result_ex_1.png)


![final_result_2](/output_images/result_ex_2.png)


![final_result_3](/output_images/result_ex_3.png)


![final_result_4](/output_images/result_ex_4.png)


![final_result_5](/output_images/result_ex_5.png)


![final_result_6](/output_images/result_ex_6.png)


#### Apply pipeline to video

Apply the pipeline to each frame in project video and write final solution video in P5_solution_v6.mp4. Code can be found in code cell No. 22, 23 in P5.ipynb

Final result video can be found in ![final_video](/P5_solution_v5.mp4)

It can also be viewed on Youtube: https://youtu.be/VpdrtnY0if0

---

### Reflections

Here are some challenges and thoughts that I had in this project:

    1. Feature extraction to classify vehicle and non-vehicle. In this project, color histogram, spatial binning of colors, and histogram of oriented gradient(HOG) features were extracted and used as features to detect vehicle. There are so many image feature can be used for this purpose, so find proper features is a very challenge work. In this project, those thee types of feature used is effective but it may not be robust enough for other situation.

    2. Classifier search and training. I tried several classifier from sklearn. It looks like the linear SVM classifier introduced in course material perform best. I could get high accuracy in Rainforest classifier but it doesn't perform well in real detection in test image. I am not sure why. As another thought, convolution neural network may be used to classify vehicle which may provide better result. And it can be next step to carry this project further.

    3. False positive reducing. The accuracy of classifier is not nigh enough, only 0.9845. I would like to improve it to over 0.99. The provided training dataset is not well balanced. I found generally, it detected dark color car better than light color car in all classifier that I tested. So the classifier tends to detect shaded tree or dark area on separation block as vehicle. Playing with different sliding window size, and heat map threshold may be able to eliminate them but making the training data more balanced and generalizing classifier may be a better approach
    
    4. HOG feature extraction is computation intensive operation. Calculating HOG for whole image once and extract feature for sliding windows from the calculated feature array will be much more effetive way to extract HOG feature. That is another improvement that can be done. 

