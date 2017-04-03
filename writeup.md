##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image0]: ./writeup_data/chessboard_corners.png "Chessboard Corners"
[image1]: ./writeup_data/undist_chess.png "Chess_Undistorted"
[image2]: ./writeup_data/undist_test1.png "Test1 Transformed"
[image3]: ./writeup_data/rgb_play.png "RGB"
[image4]: ./writeup_data/hls_play.png "HLS"
[image5]: ./writeup_data/rgb_play.png  "HSV"
[image6]: ./writeup_data/ycrcb_play.png  "YCrCb"
[image7]: ./writeup_data/stack_bin_unwarp.png  "Stack"
[image8]: ./writeup_data/stack_bin_warp.png  "Stack"
[image9]: ./writeup_data/comb_bin_unwarp.png  "Stack"
[image10]: ./writeup_data/comb_bin_warp.png  "Stack"
[image11]: ./writeup_data/hist.png  "hist"
[image12]: ./writeup_data/ptransform.png  "ptransform"
[image13]: ./writeup_data/poly.png  "poly"
[image14]: ./writeup_data/radius.png  "radius"
[image15]: ./writeup_data/plot_lane.png  "plot_lane"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Writeup has been provided as writeup.md
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Reference to the camera calibration code : [Camera Calibration Code](https://github.com/udacity/CarND-Camera-Calibration)

```python
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.4

img_calib = glob.glob('./camera_cal/*')

for fname in img_calib:
    img = cv2.cvtColor(cv2.imread(fname.strip()), cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(img, (nx,ny), None)
    # if found, add object points, image points
    if ret==True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # draw and display corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

# camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
```

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
![alt text][image0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.

```python
def undistort(img, mtx, dist):
    img_udist = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)
    ax2.imshow(img_udist)
    ax2.set_title('Undistorted Image', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return img_udist
```

```python
img = cv2.imread('./camera_cal/calibration3.jpg')
img = undistort(img, mtx, dist)
```

![alt text][image1]



###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
The first step was to explore the different color spaces. This is the most important step because if the lane information is not captured robustly in a color space, then the rest of the pipeline (e.g. threshold, gradients) will fail. It was observed that the Saturation (S) color channel of the HLS color space captured both the yellow and white lane marking.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

I used a combination of color and gradient thresholds to generate a binary image. The function below accepts different values of thresholds and a flag (SOBEL) to decide whether to use gradient threshold or color threshold:

```python
def color_binary(img_channel, c_tmin=175, c_tmax=255, sx_tmin=20, sx_tmax=100, SOBEL=False):
    binary = np.zeros_like(img_channel)
    if SOBEL==True:
        sobelx = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        binary[(scaled_sobel >= sx_tmin) & (scaled_sobel <= sx_tmax)] = 1
        return binary
    else:
        binary[(img_channel >= c_tmin) & (img_channel <= c_tmax)] = 1
        return binary
```

The function is used twice on the same image, once with SOBEL=TRUE and once with SOBEL=FALSE:

```python
sxbinary = color_binary(img_roi[:,:,2], c_tmin=175, c_tmax=255, sx_tmin=20, sx_tmax=100, SOBEL=True)

cbinary = color_binary(img_roi[:,:,2], c_tmin=175, c_tmax=255, sx_tmin=20, sx_tmax=100, SOBEL=False)
```

The two outputs are stacked for visualization:

```python
stack_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, cbinary))
stack_binary=np.uint8(255*stack_binary/np.max(stack_binary))
```
![alt text][image7]
![alt text][image8]
And then combined for further processing:

```python
# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(cbinary == 1) | (sxbinary == 1)] = 1
```
![alt text][image9]
![alt text][image10]

At this point, I have a binary image of lane markings as viewed from the top. We plot a histogram of the image along the horizontal direction to locate the position of the lane markings. The position of the lane markings as obtained from the histogram are then used in the next step for the sliding windows.
```python
# Histogram
histogram = np.sum(combined_binary[:,:], axis=0)
```
![alt text][image11]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The source and destination points were hardcoded as follows:

```python
src = np.float32([[1030, 660], [740, 480], [550, 480], [275, 660]])
dst = np.float32([[1050, 700], [1050, 0], [270, 0] , [270, 700]])
```

```python
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

img_roi = cv2.warpPerspective(img_roi, M, (img.shape[1], img.shape[0] ), flags=cv2.INTER_LINEAR)
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image12]



####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

A sliding window was used to scan for lane markings in the image. The search started at the bottom of the image where the histogram peaks were, and moved vertically while maintaining a margin of +/-100 pixels. The output of the sliding windows was x and y points. Using polyfit(), the coefficients of the left-right polynomial were obtained:

```python
left_coefficients.append(np.polyfit(lefty, leftx, 2))
right_coefficients.append(np.polyfit(righty, rightx, 2))
```

And then a Quadratic Equation was plotted based on the coefficients:

```python
# Generate x and y values for plotting
## Quadratic Equation: ay2+by+c
ploty = np.linspace(0, combined_binary.shape[0]-1, combined_binary.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

![alt text][image13]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

![alt text][image14]

```python
# Radius is calculated at the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

An example of the result plotted back down onto the road with the lane markings:

![alt text][image15]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Choosing the correct color space.
2. Choosing the correct thresholds for color and binary
3. Deciding the number of previous frames to average across

The pipeline is likely to fail if another vehicle is in front of our vehicle. This could possibly give an incorrect histogram output. One way is to detect the presence of a vehicle in front and change the strategy accordingly. 

