#cv2 is a library this is opencv
import cv2
#dlib is required to detect landmarks
import dlib
#get the indexes of the array from the dictionary
from imutils import face_utils
#to calculate distance between the eyes
from scipy.spatial import distance as dist
#array library in python
import numpy as np
#cascade classifier locations should be inside the folder
Cascade_Face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Cascade_Eye = cv2.CascadeClassifier('haarcascade_eye.xml')
Cascade_Smile = cv2.CascadeClassifier('haarcascade_smile.xml')
#The shape_predictor_68_face_landmarks.dat file is the pre-trained Dlib model for face 
Predictor_Path = 'shape_predictor_68_face_landmarks.dat_2'
#to get the face detector and the predictor working
FaceDetector = dlib.get_frontal_face_detector()
FacePredictor = dlib.shape_predictor(Predictor_Path)

#get the indexes using the face utils package and find the eyes locations
#42-48
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#36-42
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
font = cv2.FONT_HERSHEY_SIMPLEX

#if falls below 0.3 counter += 1
EYE_THRESHOLD_VALUE = 0.2
#when counter of frames reaches 3 = 1 blink
EYE_FRAMELIMIT = 3
#frame counter
COUNTER = 0
#total blink counter
TOTAL = 0

#initialize web cam for reading 0 is the current webcam
CaptureFromWebCam = cv2.VideoCapture(0)

def Calculate_EAR(CapturedEye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(CapturedEye[1], CapturedEye[5])
    B = dist.euclidean(CapturedEye[2], CapturedEye[4])
    #horizontal landmark
    C = dist.euclidean(CapturedEye[0], CapturedEye[3])
    ear = (A + B) / (2.0 * C)
    #print (A)
    #print("Ablove is A")
    #print (B)
    #print("Ablove is B")
    #print (C)
    #print("Ablove is C")
    #print(ear)
    #print("Ablove is The EAR")
    return ear

def FaceShape_To_Coordinates(shape, dtype="int"):
    #create an array of 68 rows and 2 columns for x and y coordinates
    FaceCoordinates = np.zeros((68,2), dtype=dtype)
    for i in range(0,68):
        FaceCoordinates[i] = (shape.part(i).x, shape.part(i).y)
    return FaceCoordinates

def FaceSmileEyeDetect():
    #FACE DETECTION        
    FacesFromCam = Cascade_Face.detectMultiScale(GrayFrame, 1.3, 5)
    for (xCoordinate,yCoordinate,widthFace,heightFace) in FacesFromCam:
        #draw rectangle on face
        cv2.putText(Frame, "Face Detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.rectangle(Frame,(xCoordinate,yCoordinate),(xCoordinate+widthFace,yCoordinate+heightFace),(0,0,255),5)
        
        RegionOfInterest_Gray = GrayFrame[yCoordinate:yCoordinate + heightFace, xCoordinate:xCoordinate + widthFace]
        RegionOfInterest_Color = Frame[yCoordinate:yCoordinate + heightFace, xCoordinate:xCoordinate + widthFace]
        
        #EYE DETECTION
        DetectedEyes = Cascade_Eye.detectMultiScale(RegionOfInterest_Gray, 1.3, 5)
        for(E_xCoordinate, E_yCoordinate, E_width, E_height) in DetectedEyes:
            cv2.putText(Frame, "Eyes detected", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.rectangle(RegionOfInterest_Color,(E_xCoordinate,E_yCoordinate), (E_xCoordinate + E_width, E_yCoordinate + E_height),(0,0,255),2)
        
        #SMILE DETECTION
        DetectedSmile = Cascade_Smile.detectMultiScale(RegionOfInterest_Gray, 1.5, 15)
        for(S_xCoordinate, S_yCoordinate, S_width, S_height) in DetectedSmile:
            cv2.putText(Frame, "Smile life sign detected: Not a SPOOF", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.rectangle(RegionOfInterest_Color,(S_xCoordinate,S_yCoordinate), (S_xCoordinate + S_width, S_yCoordinate + S_height),(0,0,255),2)

while 1:
    #return a bool value or frame from web cam
    ReturnVal, Frame = CaptureFromWebCam.read()
    if ReturnVal == False:
        print('Web camera capture failure detected please check camera!!!')
        break
    #Frame = imutils.resize(Frame, width=450)
    GrayFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    Faces = FaceDetector(GrayFrame, 0)
    for face in Faces:
        #apply facial landmark detector and returns the shape of the face
        FaceShape = FacePredictor(GrayFrame, face)
        #Convert the shape to a NumPy Array
        FaceShape = FaceShape_To_Coordinates(FaceShape)
        #loop the NumPy Array to track the facial landmarks
        for (xCoordinate, yCoordinate) in FaceShape:
            cv2.circle(Frame, (xCoordinate, yCoordinate), 1, (0, 255, 0), -1)
        
        #when the coordinates hit the values in the dictionary we can locate the eyes we have real time coordinates of the left and right eye now
        leftEye = FaceShape[lStart:lEnd]
        rightEye = FaceShape[rStart:rEnd]
        
        #calculate EAR for both eyes
        leftEAR = Calculate_EAR(leftEye)
        rightEAR = Calculate_EAR(rightEye)
        
        #Calculate EAR for both eyes and divide by 2 
        ear = (leftEAR + rightEAR) / 2.0
        
        #Convex is an object with no interior angles, hull is the exterior or shape of the object
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        #contours are drawn over the convex hull witht the coordinates given
        cv2.drawContours(Frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(Frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        if TOTAL < 3:
            cv2.putText(Frame, "SPOOF", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        #check to see if EAR is below the threshold 0.3
        if ear < EYE_THRESHOLD_VALUE:
            #increase the counter
            COUNTER += 1
        else:
            #frames where ear is less than 0.3
            #keeps increasing counter
            #if the frame counter is greateer than 3 
            #or an eye blink lasted for greater than 3 frames
            if COUNTER >= EYE_FRAMELIMIT:
                #increase the total counter
                TOTAL += 1              
                #one blink recorded now reset the counter
                COUNTER = 0
        if TOTAL > 3:
            cv2.putText(Frame, " NOT A SPOOF", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
          
        cv2.putText(Frame, "Blinks: {}".format(TOTAL), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(Frame, "EAR: {:.2f}".format(ear), (300, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
        #Detect face eye and smile
        FaceSmileEyeDetect()
            
    cv2.imshow('Anti Spoof FaceDetector',Frame)
    k = cv2.waitKey(30) & 0xff
    #escape key
    if k == 27:
        break
#stop capturing from web cam
CaptureFromWebCam.release()
#close window
cv2.destroyAllWindows()