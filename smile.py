import cv2
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
def  detection(grayscale,image):
    face=face_cascade.detectMultiScale(grayscale,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        reg_gray=grayscale[y:y+h,x:x+h]
        reg_color=image[y:y+h,x:x+w]
        eye=eye_cascade.detectMultiScale(reg_gray,1.2,18)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(reg_color,(ex,ey),(ex+ew,ey+eh),(0,180,60),2)
        smile=smile_cascade.detectMultiScale(reg_gray,1.7,20)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(reg_color,(sx,sy),(sx+sw,sy+sh),(255,0,130),2)
    return image
video_capture=cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    _,image=video_capture.read()
    grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canvas=detection(grayscale,image)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) and 0xFF==ord('q'):
    # if cv2.waitKey(1) and input()=='q':
        break
video_capture.release()
cv2.destroyAllWindows() 

