"""
Face_Counter v1.4
Back to status system with dlib
This method uses SIGNIFICANTLY LESS cpu power
accuracy is lower than v1.3 tho

1.4.1 updates
-------------
config.cfg files added. For that time, only program title and camera adress is active on cfg.


1.4.2 updates
-------------
"configparser" library added, now we can parse any parameter from the cfg file without a problem.
Also now we can use only one config file for multiple .exe


1.4.3 updates
-------------
nearly all adjustable parameters are available in cfg file.
also I added delay between frames in videostream loop. So the program runs slower,
and it uses less CPU power.


2021.10.7 updates
-----------------
Title on the left top corner removed.
Now, program writes the data to the log file "in real time".
Now, program runs without windows cmd, only its camera window.
Texts on the screen are translated to turkish.



Future Plans
-----------
This code will differentiate the body and the faces.
We have to find another caffemodel online or
train by ourselves. If we can run 2 trained model algorithms
simultaneously, we can count people and faces separately and
find the number of people that looked at the ad sign.

Alternative Solution: In opencv and dlib, we can process
human eyes and their angles. By using that, we can process
for ad sign. But I have no idea how to do that. This is
just a theoretical idea (It's possible).

Also, I will try to implement an SQL system in this program.
But if I fail, we can run another program to pass the data
to the SQL server from the Camera Log txt file.




-Barış Gökmen
"""
# import necessary packages
from typing import NewType
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from configparser import ConfigParser
from imutils.video import VideoStream
from imutils.video import FPS
from datetime import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2




# initializing config parse for loading config data to program
config = ConfigParser()
config.read("sample_config.cfg")


camera_name = "camera 1"
#loading data from cfg file
title_left_top = config.get(camera_name,"name")
camera_string = config.get(camera_name,"string")
log_file_name = config.get(camera_name,"log_file_name")
camera_input = int(config.get(camera_name,"Demo"))

video_width = int(config.get("settings","frame_width"))
screen_prints = int(config.get("settings","screen_prints"))

confidence_level =float(config.get("program_parameters","confidence"))
maxDisappeared = int(config.get("program_parameters","maxDisappeared"))
maxDistance = int(config.get("program_parameters","maxDistance"))
skip_frames = int(config.get("program_parameters","skip_frames"))
delay_between_frames = float(config.get("program_parameters","delay_between_frames"))




# initializing the file stream for recording the ID entry date and time data.
f = open(log_file_name, "w", buffering=1)
f.write("FACE COUNTER SYSTEM LOG\n")
f.write("Camera: {}\n".format(title_left_top))


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q","--prototxt", default= "deploy.prototxt",
	help="prototxt input")
ap.add_argument("-y","--model", default= "mydlfile.caffemodel",
	help="caffemodel input")
args = vars(ap.parse_args())




# initialize the list of class labels MobileNet SSD was trained to detect
# load our serialized model from disk
# my model for "FACE DETECTION"
print("[INFO] loading model...")
mynet = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# initialize the video stream from camera
print("[INFO] starting video stream...")
if camera_input == 0:
	vs = VideoStream(src=camera_string, framerate=1).start() # IP camera
else:
	vs = VideoStream(src=0).start()	# local camera on system (webcam)
time.sleep(1.0)


# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None


# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared, maxDistance)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
total_face = 0

# start the frames per second throughput estimator
fps = FPS().start()



# loop over frames from the video stream

while True:
	
	# grab the next frame from Video Stream
	frame = vs.read()
	time.sleep(delay_between_frames)
	#frame = cv2.flip(frame,1) #this is for mirroring the camera (NOT NECESSARY)
	

	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=video_width)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]


	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Bekliyor"
	rects = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % skip_frames == 0:
		# set the status and initialize our new set of object trackers
		status = "Taraniyor"
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections

		#my blop part
		myblob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		mynet.setInput(myblob)
		mydetections = mynet.forward()


		# loop over the detections
		for i in np.arange(0, mydetections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			myconfidence = mydetections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if myconfidence > confidence_level:

				# extract the index of the class label from the
				# detections list
				idx = int(mydetections[0, 0, i, 1])

				# compute the (x, y)-coordinates of the bounding box
				# for the object
				mybox = mydetections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = mybox.astype("int")

				if screen_prints == 1:
					cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Izleniyor"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			if screen_prints == 1:
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))


	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)
	total = 0
	# loop over the tracked objects
	for (objectID, centroid,) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# printing ID to the screen
		if screen_prints == 1:
			text = "ID {}".format(objectID+1)
			cv2.putText(frame, text, (centroid[0]-50, centroid[1]-60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		# this was the dot for centroid
		# cv2.circle(frame, (centroid[0], centroid[1]-40), 4, (0, 255, 0), -1) 

		if objectID+1 > total_face:
			total_face = objectID+1
			#updating the log file with new ID entry
			f.write("\nID {}    ->    ".format(objectID+1))
			f.write(str(datetime.now()))

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Toplam Kisi", total_face),
		("Durum", status),
	]

	if screen_prints == 1:
		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
			#printing title on the left top area of the frame
			#cv2.putText(frame, title_left_top, (15,30),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)


            
	# show the output frame
	
	cv2.imshow(title_left_top, frame)
	key = cv2.waitKey(1) & 0xFF
	print_key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q") or key == ord("Q"):
		break

	if print_key == ord("p") or print_key == ord("P"):
		screen_prints = (screen_prints + 1) % 2


	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[RESULT] total counted faces: {}".format(total_face))

#stop the video stream from camera
vs.stop()

# stop the file stream for log
f.write("\n\nTotal Counted Faces: {}".format(total_face))
f.close()

# close any open windows
cv2.destroyAllWindows()