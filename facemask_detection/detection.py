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
from scipy.signal import convolve2d

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
	
def skin_detection(image):
	"""Use a color mask to isolate skin in normal light image"""
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	blue  = image[:,:,0].astype(numpy.int16)
	green = image[:,:,1].astype(numpy.int16)
	red   = image[:,:,2].astype(numpy.int16)

	mask = (red > 96) & (green > 40) & (blue > 10) &            \
       	((image.max() - image.min()) > 15) &                    \
       	(numpy.abs(red - green) > 15) &                         \
       	(red > green) &                                         \
   		(red > blue)

	skin = image * mask.reshape(mask.shape[0], mask.shape[1], 1)
	skin = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
	
	return skin


# Collect and sanitize input images
## get image
wOut_mask = cv2.imread('pp1.png')
with_mask = cv2.imread('pp2.png')

## change image's color format
with_mask = cv2.cvtColor(with_mask, cv2.COLOR_BGR2RGB)
wOut_mask = cv2.cvtColor(wOut_mask, cv2.COLOR_BGR2RGB)

show(wOut_mask, "Without mask")
show(with_mask, "With mask")

# Cut out head and neck region out of images
## use Haar cascade provided by professor Feng
cascade_file = "haarcascade_frontalface_default.xml"
wOut_mask = isolate_face(wOut_mask, cascade_file)
with_mask = isolate_face(with_mask, cascade_file)

w, h = min(wOut_mask.shape[:2], with_mask.shape[:2]) 

wOut_mask = cv2.resize(wOut_mask, (h, w))
with_mask = cv2.resize(with_mask, (h, w))

show(wOut_mask, "Without mask")
show(with_mask, "With mask")


# Detect the skin area of the new images
## consult code from homework 3    
skin_wOut_mask = skin_detection(wOut_mask)
skin_with_mask = skin_detection(with_mask)

show(skin_wOut_mask, "Skin without mask")
show(skin_with_mask, "skin with mask")


# Replace the mask region of that image with the corresponding 
# skin region of the other image
## take the difference between the two skin images
removed_mask = numpy.zeros((w, h, 3), dtype=int)

show(with_mask, "With mask again")
for i in range(h):
	for j in range(w):
		if (skin_with_mask[j,i,0] == 0):
			removed_mask[j,i,0] = wOut_mask[j,i,0] 
			removed_mask[j,i,1] = wOut_mask[j,i,1] 
			removed_mask[j,i,2] = wOut_mask[j,i,2] 
		else:
			removed_mask[j,i,0] = with_mask[j,i,0] 
			removed_mask[j,i,1] = with_mask[j,i,1] 
			removed_mask[j,i,2] = with_mask[j,i,2] 


show(removed_mask, "Merged images")

# Repair any evidence of the image crossover
## do some smoothing perhaps
gamma = removed_mask / 255
gamma_image = gamma ** 1.3

show(gamma_image, "Gamma transform")
	
kernel = numpy.zeros((5, 5))
kernel += 1/25

# smoothed = convolve2d(removed_mas, -1, kernel)
# show(smoothed, "Smoothed")

# Enhance the image
## choose a couple techniques from homework 2


