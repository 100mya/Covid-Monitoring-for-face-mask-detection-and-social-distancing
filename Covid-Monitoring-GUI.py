from tkinter import *
#from tkinter.ttk import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import filedialog
from functools import partial


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


def sdd(video_path):

	cap = cv2.VideoCapture(video_path)
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

	    #create the red boxes
	    for i in indices_r:
	        i = i[0]
	        if ci_r[i] == 0:
	        	box = boxes_r[i]
	        	cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 255), 2)

	    #create the green boxes
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
  
  
  
  
import numpy as np
import os
import cv2
import imutils
import matplotlib.pyplot as plt

with_mask = []
without_mask = []

mypath_mask = "dataset/with_mask"
for i in os.listdir(mypath_mask):
    p = os.path.join(mypath_mask,i)
    f = cv2.imread(p,0)
    f = cv2.resize(f,(100,100))
    with_mask.append(list(f))
    
mypath_nomask = "dataset/without_mask"
for i in os.listdir(mypath_nomask):
    p = os.path.join(mypath_nomask,i)
    f = cv2.imread(p,0)
    f = cv2.resize(f,(100,100))
    without_mask.append(list(f))


m_shape = len(with_mask)
nm_shape = len(without_mask)

m_array = np.empty(m_shape, dtype=np.int)
m_array.fill(1)

nm_array = np.empty(nm_shape, dtype=np.int)
nm_array.fill(0)


features = with_mask + without_mask
label = np.concatenate((m_array,nm_array))
#plt.imshow(features[0])

from sklearn.utils import shuffle
X = np.asarray(features)
Y = label
X, Y = shuffle(X, Y, random_state=42)


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()


RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)


xtrain = xtrain.reshape(1100, 10000)
xtest = xtest.reshape(276, 10000)
clf.fit(xtrain, ytrain)


from sklearn.metrics import accuracy_score
preds = clf.predict(xtest)


print("Accuracy:", accuracy_score(ytest,preds)*100)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, preds)
print(cm)


from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score

precision = precision_score(ytest, preds)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(ytest, preds)
print('Recall: %f' % recall)

# f1: tp / (tp + fp + fn)
f1 = f1_score(ytest, preds)
print('F1 score: %f' % f1)


def imgdect(path):
    tst2 = path
    img = cv2.imread(tst2,0)
    plt.imshow(img)
    img = cv2.resize(img,(10000,1))
    if clf.predict(img)== [0]:
        return 'Without mask'
    else:
        return 'With mask'


def play():

    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frm = frame[:, :, 0]
        img = cv2.resize(frm,(10000,1))
        img = img.reshape(1,10000)
        pr = clf.predict(img)
        if (pr[0]==1):
            cv2.putText(frame,'with mask',(100,450), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),1)
        else:
            cv2.putText(frame,'without mask',(100,450), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),1)
        try:
            cv2.imshow('Video', frame)
        except:
            continue
        if cv2.waitKey(1) == 13:
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()






def win_1():

	window1 = Tk()
	window1.title("Covid Monitoring")
	window1.geometry('850x500')
	window1.configure( bg = 'deep sky blue')

	frame_head = Frame(window1)
	frame_head.pack(expand = True)

	ttl = ['C', 'O', 'V', 'I', 'D', '-', 'M', 'O', 'N', 'I', 'T', 'O', 'R', 'I', 'N', 'G']
	lables = []
	for i in range(16):
		label = Label(frame_head, text=ttl[i], font=("Arial Bold", 15), foreground = 'black', background = 'white', width = 2)
		label.configure(anchor = CENTER)
		label.pack(side = LEFT)
		label2 = Label(frame_head, text=' ', font=("Arial Bold", 15), background = 'deep sky blue')
		label2.pack(side = LEFT)
		lables.append(label)

	frame2 = Frame(window1, height = 30, bg = 'white')
	frame2.pack( fill = BOTH, expand = True)

	frame = Frame(frame2)
	frame.pack(expand = True)

	button_FMD = Button(frame, text="Face Mask Detection",  font=("Arial Bold", 12), width = 50, bg = 'deep sky blue', fg = 'white', command = win_fmd)
	button_FMD.pack()

	button_SDD = Button(frame, text="Social Distancing Detection", font=("Arial Bold", 12), width = 50, bg = 'deep sky blue', fg = 'white', command = win_sdd)
	button_SDD.pack()

	window1.mainloop()


def win_fmd():

	window_fmd = Tk()
	window_fmd.title("Add New Record")
	window_fmd.geometry('950x650')
	window_fmd.configure( bg = 'white')

	frm_fmd = Frame(window_fmd, bg = 'white')
	frm_fmd.pack(expand = True)

	frame_fmd = Frame(frm_fmd, bg = 'white')
	frame_fmd.pack(expand = True)


	from tkinter import filedialog


	def c_open_file_old():
	    rep = filedialog.askopenfilenames(
	    	parent=window_fmd,
	    	initialdir='/',
	    	initialfile='tmp',
	    	filetypes=[("All files", "*")])
	    print(rep[0])
	    file = rep[0]
	    lbl_file = Label(window, text=file, font=("Arial", 15))
	    lbl_file.grid(column=25, row=5)

	# Heading Label

	lbl_Heading = Label(frame_fmd, text="Face mask detection on image", font=("Arial Bold", 26), bg = 'white')

	lbl_Heading.grid(column=1, row=0)



	lbl_browse = Label(frame_fmd, text="Select file :", font=("Arial Bold", 20), bg = 'white')

	lbl_browse.grid(column=0, row=5)

	browse_button = Button(frame_fmd, text="Open files", command=c_open_file_old)

	browse_button.grid(row=5, column=20, padx=4, pady=4)

	button_DR = Button(frame_fmd, text="Detect image", font=("Arial Bold", 12))
	button_DR.grid(row=7, column=20, padx=4, pady=4)

	a1 = Entry(frame_fmd, width = 20, font=("Arial Bold", 20), fg = 'deep sky blue')
	a1.grid(row = 8,column = 1)

	a1.insert(INSERT, 'Answer')#imgdect(str(rep[0]))

	lbl_H = Label(frame_fmd, text="\n\n", font=("Arial Bold", 26), bg = 'white')

	lbl_H.grid(row = 9, column=1)



	frame_fmdd = Frame(frm_fmd, bg = 'white')
	frame_fmdd.pack(expand = True)


	# Heading Label

	lbl_Heading2 = Label(frame_fmdd, text="Face mask detection in video stream", font=("Arial Bold", 26), bg = 'white')

	lbl_Heading2.grid(column=1, row=0)
	

	button_DRr = Button(frame_fmdd, text="Start stream and detect", font=("Arial Bold", 12))
	button_DRr.grid(row=7, column=1, padx=4, pady=4)




def win_sdd():

	window_anr = Tk()
	window_anr.title("Social Distancing Detector")
	window_anr.geometry('780x550')
	window_anr.configure( bg = 'deep sky blue')
	
	f = Frame(window_anr, bg = 'white')
	f.pack(expand = True, ipadx = 80, ipady = 80)

	frame_anr1 = Frame(f, bg = 'white')
	frame_anr1.pack(expand = True)

	fn = Frame(f, bg = 'white')
	fn.pack(expand = True)

	fn2 = Frame(f, bg = 'white')
	fn2.pack(expand = True)

	frame_anr2 = Frame(f, bg = 'white')
	frame_anr2.pack(expand = True)

	lbl_ex2 = Label(f ,text = "", font=("Arial Bold", 8), bg = 'white', fg = 'white')
	lbl_ex2.pack(expand = True)


	from tkinter import filedialog
	vp = []

	def run():
		sdd(vp[0])

	def live():
		video_capture = cv2.VideoCapture(0)
		while True:
		    # Capture frame-by-frame
		    ret, frame = video_capture.read()
		    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		    frm = frame[:, :, 0]
		    img = cv2.resize(frm,(10000,1))
		    img = img.reshape(1,10000)
		    try:
		        cv2.imshow('Video', frame)
		    except:
		        continue
		    if cv2.waitKey(1) == 13:
		        break

		video_capture.release()
		cv2.destroyAllWindows()


	def c_open_file_old():
	    rep2 = filedialog.askopenfilenames(
	    	parent=window_anr,
	    	initialdir='/',
	    	initialfile='tmp',
	    	filetypes=[("All files", "*")])
	    print(rep2[0])
	    file = rep2[0]
	    vp.append(file)
	    lbl_file = Label(fn2, text='file : ' + file, font=("Arial", 10), bg = 'white', fg = 'black')
	    lbl_file.pack(expand = True)
	    return file

	lbl_name = Label(frame_anr1 ,text = "Detection in Video", font=("Arial Bold", 20), width = 45, bg = 'deep pink', fg = 'white')
	lbl_name.grid(row = 0,column = 0)


	sbrowse_button = Button(fn, text="Select file", font=("Arial Bold", 12), bg = 'deep sky blue', fg = 'white', command=c_open_file_old)
	sbrowse_button.grid(row=0, column=0, padx = 4)

	sbutton_DR = Button(fn, text="Start Video", font=("Arial Bold", 12), bg = 'deep sky blue', fg = 'white', command = run)
	sbutton_DR.grid(row=0, column=1, padx=4)


	lbl_add = Label(frame_anr2 ,text = "Detection in Live Stream", font=("Arial Bold", 20), width = 45, bg = 'deep pink', fg = 'white')
	lbl_add.grid(row = 0,column = 0)

	lbl_ex = Label(frame_anr2 ,text = "", font=("Arial Bold", 15), bg = 'white', fg = 'white')
	lbl_ex.grid(row = 1,column = 0)

	sbutton_DRr = Button(frame_anr2, text="Start stream and detect", font=("Arial Bold", 12), bg = 'deep sky blue', fg = 'white', command = live)
	sbutton_DRr.grid(row=3, column=0, padx=4, pady=4)



win_1()
