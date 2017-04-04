## Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Dataset

In order to train the classifier you will need to download the dataset with the images of [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). After unzipping create a folder called data and copy the contents there.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points  

### Histogram of Oriented Gradients (HOG)

#### 1. Extraction of HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook named `training.ipynb`. This cell contains functions for feature extraction of images. The function responsible for HOG features extraction is called: `get_hog_features()`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car_not_car_exploration](http://i.imgur.com/VPLxYLu.png)

I then explored different color spaces and different `skimage.hog()` parameters  like(`orientations`, `pixels_per_cell`, and `cells_per_block`).  I tested  lot of random images from each of the two classes to get a feel for what the hog features looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![hog features](http://i.imgur.com/F52uW4q.png)

#### 2. Defining HOG parameters.

The final HOG parameters defined firstly by inspecting the feature images and determining which combination describes better the original image. Next i came up with the final result by validating the classifier results. 

The chosen colorspace is the `YCrCb` color space. `HSV` also provides good results but eventually at the end `YCrCb` color space helps reduce false positive detections significantly.

#### 3. Training a classifier using the selected HOG features and color features.

A linear SVM was chosen to classify this specific project. The training process can be seen in the cell id#10 in `training.ipynb`. Using all available color-space channels (0,1 and 2) along with the HOG features, spatial features and color histograms, the accuracy of the classifier reached 0.99%. The train and test set consist of 80% and 20% respectively of the initial data set.

In order to improve the classifier before creating the final video i performed a grid search using the tool `GridSearchCV` in cell id#13. The parameters tested are C (Penalty parameter C of the error term) and tol (Tolerance for stopping criteria). The Improvement was minor. From 0.9882 accuracy (default settings) increased to 0.9901 (Optimized)

### Sliding Window Search

#### 1. Sliding window search.

To be able to capture all sizes of vehicles, different scales of rectangles must be apllied. A car may be bigger near the camera but smaller as it passes away. Consequently a combination of different scales is implemented. After many trials i ended up with the scales below:

Scales =[1,1.3,1.5,1.8,2,2.4,3]

The scale list resulted after validating the accuracy of the final video. The code is located in cell id#7 in `vehicle_detection.ipynb`.

For visualization purposes a reduced number of scales has been applied to the image below:

![scale_box_demonstration](http://i.imgur.com/HWQvPVK.png)

#### 2. Examples of pipeline in test images.

After searching on all the scales indicated above, using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, the results are satisfying. The test images can be shown below:

![detection_test_image_1](http://i.imgur.com/xehzEuf.jpg)
![detection_test_image_2](http://i.imgur.com/qq7KHZb.jpg)
![detection_test_image_3](http://i.imgur.com/lKF3lPu.jpg)
![detection_test_image_4](http://i.imgur.com/ZN2wcF0.jpg)
![detection_test_image_5](http://i.imgur.com/ZiAr9S2.jpg)
![detection_test_image_6](http://i.imgur.com/zjE2won.jpg)

(Code in cell id#8)

The false positives are almost obsolete and the classifier is working fine.

---

### Video Implementation

#### 1. Final video output.
Here's a link to my [video result](./project_video_result.mp4)
The pipeline is implemented in cell id#8 in process_image() function.

#### 2. Filtering for false positives and smoothing

##### First stage

For each frame i create a heatmap and then threshold it to identify vehicle positions. Next i use `scipy.ndimage.measurements.label()`to identify individual blobs in the heatmap assuming each blob corresponds to a vehicle. Finally I construct bounding boxes to cover the area of each blob detected.

##### Second stage

Each frame's boxes are saved into a buffer of specific size (n_frames=10). For every last 10 frames the boxes are combined into a heatmap and a threshold is being applied (7). The boxes that are not overlapped for at least 7 frames are omitted. This way i ensure that any false positive will be omitted since it is not persisted for consequent frames. Next i apply the same procedure as above and the final correct bounding boxes are drawn.

The function `process_bboxes()` in cell id#3 includes this functionality. In the pipeline this function is called twice. One time for every frame and one to average n_frames (cell id#9).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One major improvement would be to deal with the problem when one car is next to the other and sliding windows identify them as one. One solution would be to predict the centroids of each bounding box and when the outcome is not satisfying the prediction will take care of the bounding boxes. This means that a method should be applied and every n frames the training shall be repeated to validate predictions and adjust accordingly.

