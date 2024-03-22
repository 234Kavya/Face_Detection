import cv2


#0 means default video source computer web cam
cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(3,480)
harcascade = 'model/harcascade_frontalface_default.xml'


while True :
    ret , frame = cap.read()
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')  
  #convert image to grayscale for better detection
    img_gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) 
  # Print image shape for debugging
  # print("Grayscale image shape:", gray.shape)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img_gray , (x,y) , (x+w , y+h) , (255 ,0 ,0),2)

    cv2.imshow('Face Detection' , img_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()