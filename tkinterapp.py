# import the necessary packages
from __future__ import print_function
from photoboothapp import PhotoBoothApp
import argparse
import time
import imutils
import cv2
import threading


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output",  default='./people',
	help="path to output directory to store snapshots")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = cv2.VideoCapture(0)

# start the app
pba = PhotoBoothApp(vs, args["output"])
pba.root.mainloop()
