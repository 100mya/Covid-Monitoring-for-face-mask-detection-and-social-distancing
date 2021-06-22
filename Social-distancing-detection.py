import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from itertools import combinations
import math


classes = None
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
# read pre-trained model and config file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


cap = cv2.VideoCapture('PETS2009.avi')
while True:
    # Capture frame-by-frame
    prev_time = time.time()
    ret, frame_read = cap.read()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
    image = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    Width = image.shape[1]
    Height = image.shape[0]
    

    # create input blob 
    # set input blob for the network
    net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))

    # run inference through the network
    # and gather predictions from output layers

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []
    Width = image.shape[1]
    Height = image.shape[0]
    centeres = {}
    b = {}
    i = 0
    con = {}
    ci = {}

    #create bounding box 
    for out in outs:
    	for detection in out:
        	scores = detection[5:]
        	class_id = np.argmax(scores)
        	confidence = scores[class_id]
        	if confidence > 0.1:
        		center_x = int(detection[0] * Width)
        		center_y = int(detection[1] * Height)
        		centeres[i] = (center_x, center_y)
        		w = int(detection[2] * Width)
        		h = int(detection[3] * Height)
        		x = center_x - w / 2
        		y = center_y - h / 2
        		class_ids.append(class_id)
        		ci[i] = class_id
        		confidences.append(float(confidence))
        		boxes.append([x, y, w, h])
        		b[i] = [x, y, w, h]
        		con[i] = float(confidence)
        		i = i+1

    print(b)
    print(boxes)


    rl = []
    gl = []

    for (id1, p1), (id2, p2) in combinations(centeres.items(), 2):
    	xx = p1[0] - p2[0]
    	yy = p1[1] - p2[1]
    	dis = math.sqrt(xx**2 + yy**2)
    	if dis < 25:
    		if id1 not in rl and ci[id1] == 0:
    			rl.append(id1)
    		if id2 not in rl and ci[id2] == 0:
    			rl.append(id2)
    	else:
    		if id1 not in gl and ci[id1] == 0:
    			gl.append(id1)
    		if id2 not in gl and ci[id2] == 0:
    			gl.append(id2)


    for i in rl:
    	if i in gl:
     		gl.remove(i)

    print(rl)
    print(gl)

    boxes_r = []
    boxes_g = []
    cr = []
    cg = []
    ci_r = []
    ci_g = []

    for i in rl:
    	boxes_r.append(b[i])

    for j in gl:
    	boxes_g.append(b[j])


    for i in rl:
    	cr.append(con[i])

    for j in gl:
    	cg.append(con[j])


    for i in rl:
    	ci_r.append(ci[i])

    for j in gl:
    	ci_g.append(ci[j])

    print(boxes_r)
    print(boxes_g)


    indices_r = cv2.dnn.NMSBoxes(boxes_r, cr, 0.1, 0.1)
    indices_g = cv2.dnn.NMSBoxes(boxes_g, cg, 0.1, 0.1)

    #check if is people detection
    for i in indices_r:
        i = i[0]
        if ci_r[i] == 0:
        	box = boxes_r[i]
        	cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 255), 2)

    for i in indices_g:
        i = i[0]
        if ci_g[i] == 0:
        	box = boxes_g[i]
        	cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 255, 0), 2)

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(1/(time.time()-prev_time))
    cv2.imshow('Demo', image)
    cv2.waitKey(3)

cap.release()
cv2.destroyAllWindows()
