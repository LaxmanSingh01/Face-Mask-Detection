import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    points=detector.detectMultiScale(gray)
    for (x, y, w, h) in points:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow('face',frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()