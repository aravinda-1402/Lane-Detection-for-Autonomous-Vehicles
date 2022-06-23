import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.misc import imresize

###############################################################################################################################

def cameracalib():
 # termination criteria
 criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

 # prepare object points
 objp = np.zeros((6*8,3), np.float32)
 objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

 # Arrays to store object points and image points from all the images
 objpoints = [] # 3d point in real world space
 imgpoints = [] # 2d points in image plane

 images = glob.glob('/home/ghosh/Documents/*.jpg')
 for fname in images:
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  sh=gray.shape[::-1]

  #Find the chess board corners
  ret, corners = cv2.findChessboardCorners(gray, (6,8),None)

  #If found, add object points, image points (after refining them)
  if ret == True:
   objpoints.append(objp)

   #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
   imgpoints.append(corners)

   # Draw and display the corners
   img = cv2.drawChessboardCorners(img, (6,8), corners,ret)
   #cv2.imshow('img',img)
   #cv2.waitKey(1000)
        
 return objpoints, imgpoints, sh

###################################################################################################################################

def edgedetection(lane_drawn):

 img=cv2.cvtColor(lane_drawn, cv2.COLOR_BGR2GRAY)
 img=np.uint8(img)
 img=cv2.resize(img,(480,240)) 
 blur = cv2.GaussianBlur(img,(5,5),0)
 canny=cv2.Canny(blur,100,200)
 erosion = cv2.erode(canny,(5,5),iterations = 1)
 dilation = cv2.dilate(erosion,(5,5),iterations = 3)

 roi=dilation[120:240,0:480]
 #cv2.imshow("roi",roi)
 
 pts1 = np.float32([[10,120],[450,120],[0,0],[480,0]])
 pts2 = np.float32([[175,120],[230,120],[0,0],[480,0]])

 M = cv2.getPerspectiveTransform(pts1,pts2)

 dst = cv2.warpPerspective(roi,M,(480,120))
 dst = dst.astype(np.uint8)
 cv2.imshow("transform",dst)
 #print('image dtype ',dst.dtype)
 return dst
#######################################################################################################################################

def startpixels(wpixel, pic):
 for ly in range(0,len(wpixel)-1):
  if wpixel[ly]!=0:
   break

 for lx in range(0,120):
  if pic[lx,ly]!=0:
   break

 for ry in range(len(wpixel)-1,-1,-1):
  if wpixel[ry]!=0:
   break

 for rx in range(0,120):
  if pic[rx,ry]!=0:
   break

 return ly,lx,ry,rx
   
#sliding window up to down
def slidingwindowdown( x, y, pic, colourpic):  
 recpx=[] 
 recpy=[]

 while(True):
  cv2.rectangle(colourpic,(y-10,x),(y+10,x+10),(0,255,0),1)

  avgi=0
  avgj=0
  c=0
  for i in range(x, x+10):
   for j in range(y-10,y+10):
    if pic[i,j]!=0:
     c=c+1
     avgi=avgi+i
     avgj=avgj+j
     recpx.append(i)
     recpy.append(j)

  if c==0:
   break
  
  newx=(avgi//c)+1
  newy=(avgj//c)
  if newx+10>120:
   break

  x=newx
  y=newy
 
 return recpx, recpy, colourpic
  
#sliding window down to up
def slidingwindowdup( x, y, pic, colourpic): 
 recpx=[]
 recpy=[] 
 x=int(x)
 y=int(y)

 while(True):
  cv2.rectangle(colourpic,(y-10,x-10),(y+10,x),(0,255,0),1)

  avgi=0
  avgj=0
  c=0
  for i in range(x-10,x):
   for j in range(y-10,y+10):
    if pic[i,j]!=0:
     c=c+1
     avgi=avgi+i
     avgj=avgj+j
     recpx.append(i)
     recpy.append(j)

  if c==0:
   break

  newx=(avgi//c)-1
  newy=(avgj//c)

  x=newx
  y=newy
 
 return recpx, recpy, colourpic

#histogram
def histo(dst): 
 c=0
 wpixel=[]
 rows, columns=dst.shape
 #print("rows="+str(rows))
 #print("columns="+str(columns))

 for j in range(0,columns):
  for i in range(0, rows):
   if dst[i,j]!=0:
    #print("["+str(i)+","+str(j)+"]="+str(dst[i,j]))
    c=c+1
  wpixel.append(c)
  c=0 

 #plt.subplot(211),plt.imshow(dst)
 #plt.subplot(212),plt.plot(wpixel)
 #plt.show()
 return wpixel

def slidewindow(wpixel):

 ly,lx,ry,rx = startpixels(wpixel, dst)
 #print("Left lane start point="+str(lx)+","+str(ly))
 #print("Right lane start point="+str(rx)+","+str(ry))
 #print()

 colourpic1= cv2.cvtColor(dst,cv2.COLOR_GRAY2RGB)
 colourpic2= cv2.cvtColor(dst,cv2.COLOR_GRAY2RGB)
 
 #left lane pixels
 if lx<100:
  leftlanex, leftlaney, sw1pic= slidingwindowdown( lx, ly, dst, colourpic1)
 else:
  leftlanex, leftlaney, sw1pic= slidingwindowdup( lx, ly, dst, colourpic1)
 
 #print("left lane pixels: ",leftlanex,leftlaney)
 #print()


 #right lane pixels
 if rx<100:
  rightlanex, rightlaney, sw2pic= slidingwindowdown( rx, ry, dst, sw1pic)
 else:
  rightlanex, rightlaney, sw2pic= slidingwindowdup( rx, ry, dst, sw1pic)

 #print("right lane pixels: ",rightlanex,rightlaney)
 #print()
 
 #cv2.imshow("sliding window",sw2pic)   
 return leftlanex, leftlaney, rightlanex, rightlaney

#################################################################################################################################

def quadraticcurve(leftlanex, leftlaney, rightlanex, rightlaney):
 leftz = np.polyfit( leftlanex,leftlaney, 2)
 rightz = np.polyfit( rightlanex,rightlaney, 2)
 funcl=np.poly1d(leftz)
 funcr=np.poly1d(rightz)
 #print("left lane polynomials:",leftz)
 #print("right lane polynomials:",rightz)
 #print(funcl)
 #print(funcr)
 
 xl_new = np.linspace(leftlanex[0], leftlanex[-1], 50)
 yl_new = funcl(xl_new)
 xr_new = np.linspace(rightlanex[0], rightlanex[-1], 50)
 yr_new = funcr(xr_new)

 lfp=[]
 rfp=[]
 for i in range(0,50):
  lfp.append([yl_new[i],xl_new[i]])
 for i in range(0,50):
  rfp.append([yr_new[i],xr_new[i]])
 
 lfpts=np.asarray(lfp)
 rfpts=np.asarray(rfp)
 #print(lfpts)
 #print(rfpts)
 
 cv2.polylines(colourpic2, np.int32([lfpts]),False, (255,0,0),1)
 cv2.polylines(colourpic2, np.int32([rfpts]),False, (255,0,0),1)
 #cv2.imshow("curve function",colourpic2)
 #plt.imshow(colourpic2)
 #plt.show()

 rightx=funcr(5)
 leftx=funcl(5)
 offset=((rightx-center)-(center-leftx))**ym_per_pix
 
 return leftz, rightz, offset

################################################################################################################################

def radiusofcurvature(leftlanex,leftlaney,rightlanex,rightlaney,leftz,rightz):

 leftlanex = list( map(lambda x: x*xm_per_pix, leftlanex) )
 leftlaney = list( map(lambda y: y*ym_per_pix, leftlaney) )
 rightlanex = list( map(lambda x: x*xm_per_pix, rightlanex) )
 rightlaney = list( map(lambda y: y*ym_per_pix, rightlaney) ) 

 radius_l = ((1 + (2 * leftz[0] * 45 * xm_per_pix + leftz[1]) ** 2) ** 1.5) / np.absolute(2 * leftz[0])
 radius_r = ((1 + (2 * rightz[0] * 45 * xm_per_pix + rightz[1]) ** 2) ** 1.5) / np.absolute(2 * rightz[0])
 #print("left lane radius: ",radius_l)
 #print("right lane radius: ",radius_r)
 return radius_l, radius_r

###################################################################################################################################

model=load_model('/home/ghosh/Documents/full_CNN_model.h5')

xm_per_pix = 27/45  
ym_per_pix = 3.7/35

objpoints, imgpoints, sh=cameracalib()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, sh, None, None)

cap = cv2.VideoCapture(0)
while(True):
 ret, frame = cap.read()
 frame = cv2.undistort(frame, mtx, dist, None, mtx)

 small_img = imresize(frame, (80,160,3))
 small_img = np.array(small_img)
 small_img = small_img[None,:,:,:]

 pred=model.predict(small_img)[0] *255

 recent_fit.append(pred)
 if len(recent_fit)>10:
  recent_fit= recent_fit[1:]

 avg_fit=np.mean(np.array([i for i in recent_fit]), axis=0)
 blanks=np.zeros_like(avg_fit).astype(np.uint8)
 lane_drawn=np.dstack((blanks, avg_fit, blanks))
 
 dst = edgedetection(lane_drawn)
 center= dst.shape[0]/2
 wpixel = histo(dst)
 leftlanex, leftlaney, rightlanex, rightlaney = slidewindow(wpixel)
 leftz, rightz, offset = quadraticcurve(leftlanex, leftlaney, rightlanex, rightlaney)
 radius_l, radius_r = radiusofcurvature(leftlanex,leftlaney,rightlanex,rightlaney,leftz,rightz)

 imagef=imresize(frame,(720,1280,3))
 lane_image=imresize(lane_drawn, (720,1280,3))
 result=cv2.addWeighted(imagef,1,lane_image,1,0)
 
 cv2.putText(result,'Left Lane Radius:'+str(radius_l),(10,10), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
 cv2.putText(result,'Right Lane Radius:'+str(radius_r),(10,40), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255),2,cv2.LINE_AA)

 if offset<0:
  cv2.putText(result,'Offset:'+str(np.abs(offset))+'m right',(600,10), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
 elif offset>0:
  cv2.putText(result,'Offset:'+str(np.abs(offset))+'m left',(600,10), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
 else:
  cv2.putText(result,'Offset:'+str(np.abs(offset))+'m',(600,10), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)

 cv.imshow('output',result)

 #plt.subplot(221),plt.imshow(img),plt.title('Input')
 #plt.subplot(222),plt.imshow(erosion),plt.title('Edge Detection')
 #plt.subplot(223),plt.imshow(roi),plt.title('Roi') 
 #plt.subplot(224),plt.imshow(dst),plt.title('Perspective Transform') 
 #plt.show()

 if cv2.waitKey(30) & 0xFF== ord('q'):
  cv2.destroyAllWindows()

cap.release() 
##################################################################################################################################
 

