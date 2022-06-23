import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.misc import imresize

model=load_model('/media/mrmai/E086487686484F6A/ResearchEAI/full_CNN_model5_10l128f.h5')
cap = cv2.VideoCapture('/media/mrmai/E086487686484F6A/ResearchEAI/videoplayback1.mp4')

recent_fit=[]
avg_fit=[]

while(True):
 ret, frame = cap.read()

 small_img = imresize(frame, (80,160,3))
 small_img = np.array(small_img)
 small_img = small_img[None,:,:,:]

 pred=model.predict(small_img)[0] *255

 recent_fit.append(pred)
 
 if len(recent_fit)>5:
  recent_fit= recent_fit[1:]

 avg_fit=np.mean(np.array([i for i in recent_fit]), axis=0)

 blanks=np.zeros_like(avg_fit).astype(np.uint8)
 lane_drawn=np.dstack((blanks, avg_fit, blanks))
 lane_image=imresize(lane_drawn, (360,640,3))

 result=cv2.addWeighted(frame,1,lane_image,1,0)
 cv2.imshow('output',result)

 if cv2.waitKey(1) & 0xFF== ord('q'):
  cv2.destroyAllWindows()

cap.release() 


