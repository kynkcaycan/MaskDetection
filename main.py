import cv2
import time
import sqlite3
from arayuz import interfaceYap
if __name__ == '__main__' :

  interface= interfaceYap()
  kontrol=interface.interfaceyap()
  print(kontrol)
  con=sqlite3.connect("C:\sqlitedbs\ssTime.db")
  im = con.cursor()
  im.execute("""CREATE TABLE IF NOT EXISTS
      ss_Time (time)""")
  cam=cv2.VideoCapture(0)
  faceCascade=cv2.CascadeClassifier("C:\\Users\\aycan\\PycharmProjects\\maskDetection\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalcatface.xml")
  mounthCascade=cv2.CascadeClassifier("C:\\Users\\aycan\\PycharmProjects\\maskDetection\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_mcs_mouth.xml")

  fps=0
  passingTime=0

  while True:
      ret,frame=cam.read()
      frame=cv2.flip(frame,1)
      height, width, channels = frame.shape
      start_time = time.time()
      if width > 0 and height > 0:

        cv2.imshow("MASK DETECTOR",frame)
        gri_resim=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gri_resim,1.1,1)
        if len(faces)==0:
            cv2.putText(frame,"",(40,40),cv2.FONT_HERSHEY_SIMPLEX,2,(160, 100, 100),1)
        else:
           for x,y,w,h in faces:
              pt=str(fps)
              pt="FPS:"+pt[ :4]
              cv2.putText(frame, pt, (500, 410), cv2.FONT_HERSHEY_SIMPLEX, 1,(128,0,128),5, 1)
              cv2.rectangle(frame,(x,y),(x+w,y+h),(139,10,80),8)
              roi_gri=gri_resim[y:y+h,x:x+w]
              mout=mounthCascade.detectMultiScale(roi_gri,1.4,15)

              if len (mout)==0:
                  cv2.putText(frame, "Maske Var", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (128,0,128),5 ,3)
              else:
                  for x1,y1,w1,h1 in mout:
                     cv2.putText(frame, "Maske Yok", (40, 410), cv2.FONT_HERSHEY_SIMPLEX, 2, (198,0,0),5, 3)
                     cv2.rectangle(frame,(x1+x,y1+y),(x1+w1+x,y1+y+h1),(255 ,20,147),3)
                     x=x+1
                     cv2.imwrite('C:\\Users\\aycan\PycharmProjects\maskDetection\\venv\maskesizler\\'+str(x)+'resim.jpg',frame)
                     saat = (time.time())
                     im.execute("""INSERT INTO ss_Time VALUES (?)""",(saat,))
                     con.commit()
      cv2.imshow("MASK DETECTOR", frame)
      end_time=time.time()
      passingTime=end_time-start_time
      fps = 1/passingTime
      if cv2.waitKey(1) & 0xFF==ord("q"):
          break
  cam.release()
  cv2.destroyWindow()
