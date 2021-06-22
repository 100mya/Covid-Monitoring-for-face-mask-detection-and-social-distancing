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
