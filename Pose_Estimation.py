#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
import os
import gc
import pickle


def model_loader(video_side, hand_type):
    models_lst = [pickle.load(open(os.path.join("best_models", video_side+'_'+hand_type, "", model_name), 'rb')) for model_name in os.listdir(os.path.join("best_models", video_side+'_'+hand_type, ""))]
    meta_model = pickle.load(open("meta_models/"+video_side+"_"+hand_type+"_"+"meta.sav", 'rb'))
    return models_lst, meta_model

def tup_unpack(tup): return [tup[0], tup[1]]

def frames_unraveler(points_lst):
    data = None
    for i in points_lst:
        data_new = [tup_unpack(j) if j is not None else tup_unpack((np.nan,np.nan)) for j in i]
        data_new = np.array(data_new, dtype = np.float32).ravel()
        if data is None:
            data = data_new
        else:
            data = np.vstack((data, data_new))
    return data

def imputer(points_lst):
    data = frames_unraveler(points_lst)
    #import sklearn.preprocessing.Impute
    imp = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")#"mean"
    imp.fit(data); data = imp.transform(data)
    return data.reshape((1, -1)).squeeze()

def body_keypoints(video_path, video_side):
    net  = cv.dnn.readNetFromTensorflow("graph_opt.pb")
    cap = cv.VideoCapture(video_path)
    cap.set(cv.CAP_PROP_FPS, 10)
    cap.set(3, 800)
    cap.set(4, 800)
    # 'Threshold value for pose parts heat map'
    thr = 0.2#<-----------------------------------------------------------
    # 'Resize input to specific width.'
    width = 368
    # 'Resize input to specific height.'
    height = 368
    # if video is not opened
    if not cap.isOpened():
        cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open video")
    flag = True; points_lst = []; ctr = 0
    while cv.waitKey(1) < 0 and ctr < 20:
        hasFrame, frame = cap.read(); ctr += 1
        if not hasFrame:
            cv.waitKey()
            break
        #if cv.getWindowProperty('crop_frame', cv.WND_PROP_VISIBLE) < 1:
        #    break
        if cv.waitKey(10) & 0xFF == ord('q') :
            # break out of the while loop
            break
        if video_side == "left":
            crop_frame = frame[:, 0:700, :]
        else:
            crop_frame = frame[:, 700:, :]
        crop_frameWidth = crop_frame.shape[1]
        crop_frameHeight = crop_frame.shape[0]
        inp = cv.dnn.blobFromImage(crop_frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=False, crop=False)#inScale
        net.setInput(inp)
        out = net.forward()
        out = out[:, :19, :, :]
        #assert(len(BODY_PARTS) <= out.shape[1])
        points = []
        required_body_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]# see BODY_PARTS_M for reference...
        for i in required_body_points:
            # Slice heatmap of corresponding body's part.
            heatMap = out[0, i, :, :]
            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (crop_frameWidth * point[0]) / out.shape[3]
            y = (crop_frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > thr else None)
        points_lst.append(points)# not including background info.
    #destroy all windows
    cap.release()
    cv.destroyAllWindows()
    return imputer(points_lst)

def predictor(data, models_lst, meta_model):
    pred_array = np.array([model.predict_proba(data.reshape(1, -1))[0,1] for model in models_lst]).reshape(1, -1)
    #print("Pred array = ", pred_array)
    return meta_model.predict(pred_array)

def vid_classifier(video_path, video_side, hand_type):
    body_points = body_keypoints(video_path, video_side)
    models_lst, meta_model = model_loader(video_side, hand_type)
    pred_value = predictor(body_points, models_lst, meta_model)
    if pred_value:
        print("\n\nVideo of Forehand Short\n\n")
    else:
        print("\n\nVideo of Backhand Shot\n\n")


# In[37]:


video_path = input("Enter your video path (provide the absolute path if saved in other directories else provide relative path): ")
video_side = input("Is it left or right? Type (left/right): ")
hand_type = input("Is player a lefty or righty? (lefty/righty): ")
vid_classifier(video_path, video_side.lower(), hand_type.lower())
