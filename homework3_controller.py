from controller import *
import cv2
import numpy as np
import copy

def getHSV(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_arr = hsv[y,x]
        print("HSV:", hsv_arr)

robot = Robot()

print("hello")

timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

#vidCap = cv2.VideoCapture(0)

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
#leftMotor.setPosition(8.0)
#rightMotor.setPosition(16.0)





i = 0


while robot.step(timestep) != -1:
    img = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    temppic = np.frombuffer(img, np.uint8)
    out = np.reshape(temppic, (height, width, 4))
    #ret, frame = vidCap.read()
    
    avgVal = np.float32(out)
    
    #-----Tracking-----
    
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    cv2.imshow('frame', hsv)
    cv2.setMouseCallback('frame', getHSV)
    
    low = np.array([7, 90, 45])
    high = np.array([20, 220, 100])

    mask = cv2.inRange(hsv, low, high)
    hsv_filter = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    hsv_filter = cv2.dilate(hsv_filter, None, iterations = 10)
    hsv_filter = cv2.erode(hsv_filter, None, iterations = 10)
 
    cv2.imshow('hsv_filter_frame', mask)


  
    
    
    
    #-----Motion-------
    
    blur = cv2.blur(out, (5,5))

    cv2.accumulateWeighted(blur, avgVal, .5)
    absVal = cv2.convertScaleAbs(avgVal)

    diff = cv2.absdiff(absVal, out)

    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    ret,gray = cv2.threshold(gray,5,255,cv2.THRESH_BINARY)
    gray = cv2.blur(gray, (8,8))
    ret,gray = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

    gray = cv2.dilate(gray, None, iterations = 10)
    gray = cv2.erode(gray, None, iterations = 1)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, None)

    cont, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contoured = copy.deepcopy(out)
    ret,contoured  = cv2.threshold(gray,255,255,cv2.THRESH_BINARY)
    ret,contoured  = cv2.threshold(gray,255,255,cv2.THRESH_BINARY)

    cv2.drawContours(contoured, cont, -1, (255,255,255), 1)
    
    x = 0
    x2 = 0
    area = 0
    
    for cnt in cont: 
        if cv2.contourArea(cnt) < 500: 
            continue
#        else:
 #           leftMotor.setVelocity(0)
  #          leftMotor.setVelocity(0)
   #         leftMotor.setPosition(0)
    #        rightMotor.setPosition(0)

        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x2 = x + w
        y2 = y + h
        print("x: " + str(x), "y: " + str(y), "x2: " + str(x2), "y2: " + str(y2))
        
        coord = [x,x2]
        
        wid = x2 - x
        hei = y2 - y
        
        area = wid * hei
        
        print('Width: ' + str(wid) + 'Height: ' + str(hei) + 'Area: ' + str(area))
        
    if(coord[0] < 22):
            
        leftMotor.setPosition(-.7)
        rightMotor.setPosition(.7)
        print("Left")
        
    #elif(coord[0] < 52):
            
       # leftMotor.setPosition(-.7)
      #  rightMotor.setPosition(.7)
       # print("Left")
      #  break
    
    elif(coord[1] > 330):
            
        leftMotor.setPosition(.7)
        rightMotor.setPosition(-.7)
        print("Right")
        
                
   # elif(coord[1] > 300):
            
       # leftMotor.setPosition(.7)
       # rightMotor.setPosition(-.7)
      #  print("Right")
      #  break
      
    print(area)
        
    if(area < 15000):

        leftMotor.setPosition(0)
        rightMotor.setPosition(0)
        leftMotor.setPosition(float('inf'))
        rightMotor.setPosition(float('inf'))
    elif(area > 25000):

        leftMotor.setPosition(0)
        rightMotor.setPosition(0)
            
        #else:
         #   leftMotor.setPosition(float('inf'))
          #  rightMotor.setPosition(float('inf'))
    
        
#    else:
 #       leftMotor.setPosition(0)
  #      rightMotor.setPosition(0)
            
    #------Motion-------

    cv2.imshow('Contours', contoured)
    cv2.setMouseCallback('Contours', getHSV)
    cv2.imshow('Image1', out)

    
    
    
    
    
    i += 1
    
    
    
    k = cv2.waitKey(1)
    if k == 27:
        break


cv2.destroyAllWindows()
    
    
