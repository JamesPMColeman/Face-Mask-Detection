#===============================================================#
#                                                               #
#   James Coleman                                               #
#   CS 3150                                                     #
#   Project 1                                                   #
#   October 29th                                                #
#                                                               #
#===============================================================#

    #          >>>>>>>>>> Goals <<<<<<<<<<
    #   
    #   1. Cut out head and neck region out of images
    #   2. Detect the skin area of the new images
    #   3. Replace the mask region of the mask image with the 
    #   	corresponding skin region of the other image
	#	4. Repair any evidence of the image crossover
	#	5. Enhance the image
    #   
    #   

# Imports
import numpy
import cv2 
from matplotlib import pyplot

# Helper methods
def show(image, title):
    """ Helper method to display a single image 
    with pyplot """
    pyplot.figure()
    pyplot.title(title)
    pyplot.imshow(image)
    pyplot.show()

def isolate_face(image, cascade):
	""" Uses cv's CascadeClassifier to detect a face and then
	remove a border around the face."""
	face_cascade = cv2.CascadeClassifier(cascade)
	crop_wOut_mask = face_cascade.detectMultiScale(
		image,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (30, 30),
	)

	for (x, y, w, h) in crop_wOut_mask:
		cut_image = image[y - 40: y + h + 80, x: x + w, :]

	return cut_image		
	

# Collect and sanitize input images
## get image
wOut_mask = cv2.imread('pp1.png')
with_mask = cv2.imread('pp2.png')

## change image's color format
with_mask = cv2.cvtColor(with_mask, cv2.COLOR_BGR2RGB)
wOut_mask = cv2.cvtColor(wOut_mask, cv2.COLOR_BGR2RGB)
## get image dimensions?

show(wOut_mask, "Without mask")
show(with_mask, "With mask")

# Cut out head and neck region out of images
## use Haar cascade provided by professor Feng
cascade_file = "haarcascade_frontalface_default.xml"
wOut_mask = isolate_face(wOut_mask, cascade_file)
with_mask = isolate_face(with_mask, cascade_file)

show(wOut_mask, "Without mask")
show(with_mask, "With mask")


# Detect the skin area of the new images
## consult code from homework 3    


# Replace the mask region of that image with the corresponding 
# skin region of the other image
	

# Repair any evidence of the image crossover
## do some smoothing perhaps	

# Enhance the image
## choose a couple techniques from homework 2


