from __future__ import print_function

import numpy as np
import argparse
import glob
import cv2
import os 
import sys
import random

sys.path.insert(0, '/Users/developer/guru/')

from utility import basics
from fileOp.imgReader import ImageReader
from userinput.mouseOp import mouseCrop, mouseHandler
from fileOp.conf import Conf
from annotation.pascal_voc import pacasl_voc_writer

DOMINATE_CLUSTER_NO	= 2
font = cv2.FONT_HERSHEY_SIMPLEX

frame = None
videoName = None
bCropDone = False
cropStart = False
cropEnd = False
classInfo = []
blockSize = None
bCapture = False
cntFrame = 0

def cropCallback(bDone, startPos, endPos):

	global frame, videoName, bCropDone, cropStart, cropEnd, classInfo

	print('crop = {}, {}-{}'.format(bDone, startPos, endPos))
	x0 = min(startPos[0], endPos[0])
	x1 = max(startPos[0], endPos[0])
	y0 = min(startPos[1], endPos[1])
	y1 = max(startPos[1], endPos[1])

	cropStart = (x0, y0)
	cropEnd = (x1, y1)
	clone = frame.copy()
	cv2.rectangle(clone, cropStart, cropEnd, (0, 255, 0), 2)

	drawAllClass(clone)

	basics.showResizeImg(clone, videoName, 1)
	bCropDone = bDone

def drawAllClass(img):
	global classInfo

	for (name, color, start, end) in classInfo:
		if start != end:
			cv2.rectangle(img, start, end, color, 2)

def CaptureImageAndAnnotation(img):
	print('CaptureImageAndAnnotation')
	global classInfo, cntFrame

	name = 'output/img'+str(cntFrame)+'.png'
	cv2.imwrite(name,img)

	name = 'output/img'+str(cntFrame)
	voc_writer = pacasl_voc_writer(name, 'output', img.shape)
	box_w = np.amax([end[0]-start[0] for (cname, color, start, end) in classInfo])
	box_h = np.amax([end[1]-start[1] for (cname, color, start, end) in classInfo])

	print('box = {}x{}, blockSize = {}'.format(box_w, box_h, blockSize))
	box_w = ((box_w + blockSize[0] - 1) / blockSize[0]) * blockSize[0]
	box_h = ((box_h + blockSize[1] - 1) / blockSize[1]) * blockSize[1]
	print('new box = {}x{}'.format(box_w, box_h))

	for (cname, color, start, end) in classInfo:
		if (start != end):
			voc_writer.new_box(cname, (start[0], start[1], start[0]+box_w-1, start[1]+box_h-1))
	voc_writer.save()	

def main():

	global DOMINATE_CLUSTER_NO
	global frame, videoName, bCropDone, cropStart, cropEnd, classInfo, cntFrame, blockSize

	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", required=True, help="Path to video file, 'webcam' for live video from your computer camera")
	ap.add_argument("-c", "--conf", required=True,  help="json file configuration")
	ap.add_argument("-a", "--annotation", required=False, help="annotation outout path")
	ap.add_argument("-i", "--image", required=False, help="image output path")	

	args = vars(ap.parse_args())

	conf = Conf(args['conf'])
	classInfo = []
	if (conf['class'] != None):
		for name in open(conf['class']).read().split("\n"):
			color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
			classInfo.append((name, color, (0,0), (0,0)))
	
	pixels_per_cell = conf['pixels_per_cell']
	cells_per_block = conf['cells_per_block']

	blockSize = np.multiply(pixels_per_cell, cells_per_block).tolist()

	videoName = args["video"]
	vr = ImageReader(videoName, True)
	cv2.namedWindow(videoName)
	mCrop = mouseCrop(dragCallback=cropCallback)
	cv2.setMouseCallback(videoName, mouseHandler, mCrop)

	bPause = False
	bCapture = False
	while (True):

		if (bPause != True):
			ret, frame, imageName = vr.read()
			if (ret == False):
				break

			clone = frame.copy()
			drawAllClass(clone)

			key = basics.showResizeImg(clone, videoName, 10)
			if bCapture == True:
				CaptureImageAndAnnotation(frame)
			cntFrame = cntFrame + 1
		else:
			key = cv2.waitKey(10)
			if bCropDone == True:
				if key >= ord('0') and key < (ord('0') + len(classInfo)):
					classidx = key - ord('0')
					(name, color, _, _) = classInfo[classidx]
					clone = frame.copy()
					classInfo[classidx] = (name, color, cropStart, cropEnd)
					drawAllClass(clone)
					basics.showResizeImg(clone, videoName, 1)
					print('name = {} color = {} {}-{}'.format(classInfo[classidx][0], classInfo[classidx][1], classInfo[classidx][2], classInfo[classidx][3])) 
				if key == ord('d'):
					drawAllClass(clone)
					basics.showResizeImg(frame, videoName, 1)
		if key == ord('p'):
			bPause = True if (bPause == False) else False
		if key == ord('c'):
			bCapture = True if (bCapture == False) else False
		if key == ord('q'):
			break

	vr.close()
	print('program finished')

main()
