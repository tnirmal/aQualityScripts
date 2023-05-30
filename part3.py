cuda = False

import cv2
import time as t

import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import time
from pypylon import pylon
from pypylon import genicam
import sys
import subprocess
from datetime import date, time, datetime

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

                               

def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def grabimage(retry):
    results=[]
    
    today = date.today()
    #today
    #datetime.date(2020, 1, 24)
    now = datetime.now()
    #now
    #datetime.datetime(2020, 1, 24, 14, 4, 57, 10015)
    current_time = time(now.hour, now.minute, now.second)
    date_time=datetime.combine(today, current_time)
            
    good_part=0
    bad_part=0
    Quality={"Good":good_part,"Bad":bad_part,'Date':date_time}
    
    if retry == 'Low':
        try:
            t.sleep(0.25)
            # Get the transport layer factory.
            tlFactory = pylon.TlFactory.GetInstance()

            # Get all attached devices and exit application if no device is found.
            devices = tlFactory.EnumerateDevices()
            if len(devices) == 0:
                raise pylon.RuntimeException("No camera present.")

            # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
            cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
            #print(cameras)
            l = cameras.GetSize()
            print(l)
                    # Create and attach all Pylon Devices.

            images=[]

            for i, cam in enumerate(cameras):

                cam.Attach(tlFactory.CreateDevice(devices[i]))

                converter = pylon.ImageFormatConverter()

                converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
                cam.StartGrabbingMax(countOfImagesToGrab)

                while cam.IsGrabbing():
                     # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
                    grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if grabResult.GrabSucceeded() and i == 0:
                        image = converter.Convert(grabResult)
                        img = image.GetArray() 

                        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        #ori_images = [img1.copy()]
                        #display(Image.fromarray(ori_images[0]))
                        #cv2.imwrite('top',img1)
                        images.append(img1)
                    elif grabResult.GrabSucceeded() and i == 1:
                        image = converter.Convert(grabResult)
                        img = image.GetArray() 

                        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img2)
                        #ori_images = [img2.copy()]
                        #display(Image.fromarray(ori_images[0]))
                        #cv2.imwrite('right',img2)

                    elif grabResult.GrabSucceeded() and i == 2:
                        image = converter.Convert(grabResult)
                        img = image.GetArray() 

                        img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img3) 
                        #ori_images = [img3.copy()]
                        #display(Image.fromarray(ori_images[0]))

                    else:

                        print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                    grabResult.Release()
                    cameras.Close()    
            w= ['/home/algoiei/Downloads/Augmented_Data_weights-20221226T034757Z-001/Augmented_Data_weights/part3/left/best.onnx','/home/algoiei/Downloads/Augmented_Data_weights-20221226T034757Z-001/Augmented_Data_weights/part3/right/best.onnx','/home/algoiei/Downloads/Augmented_Data_weights-20221226T034757Z-001/Augmented_Data_weights/part3/top/best.onnx']

            
            names = [['nut_present','nut_missing'],['nut_present','nut_missing','spot_ok','spot_nok'],['nut_present','nut_missing','spot_ok','spot_nok']] 


            colors = [[0, 255, 0],[255, 0, 0], [0, 255, 0],[255, 0, 0]]

            for img, w, names in zip(images, w, names):
                session = ort.InferenceSession(w, providers=providers)


                #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = img.copy()
                image, ratio, dwdh = letterbox(image, auto=False)
                image = image.transpose((2, 0, 1))
                image = np.expand_dims(image, 0)
                im = image.astype(np.float32)
                im /= 255
                im.shape


                outname = [i.name for i in session.get_outputs()]
                outname

                inname = [i.name for i in session.get_inputs()]
                inname

                inp = {inname[0]:im}
                outputs = session.run(outname, inp)[0]
                ori_images = [img.copy()]
                print(outputs)
                for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
                    image = ori_images[int(batch_id)]
                    box = np.array([x0,y0,x1,y1])
                    box -= np.array(dwdh*2)
                    box /= ratio
                    box = box.round().astype(np.int32).tolist()
                    cls_id = int(cls_id)
                    score = round(float(score),3)
                    name = names[cls_id]
                    color = colors[cls_id]
                    name += ' '+str(score)
                    result=name.split(' ')
                    results.append(result[0])
                    cv2.rectangle(image,box[:2],box[2:],color,2)
                    cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.8,[225, 255, 255],thickness=2)  
                #display(Image.fromarray(ori_images[0]))
            for i in results:
                #print(i)
                if i == 'spot_nok' :
                    pin1='0'# not ok
                    pin2='1'
                    subprocess.call(["./dio4test", pin1, pin2])
                    print('NOk')
                    #god_part=+1
                    #Quality["Good"].append(str(good_part))
                    #Quality["Date"].append(date_time)
                    #return(Quality)
                    
                    break
            else: 
                pin1='1'# not ok
                pin2='0'
                subprocess.call(["./dio4test", pin1, pin2])
                #Quality["Bad"].append(str(bad_part))
                #Quality["Date"].append(date_time)
                #bad_part=+1
                #return(Quality)    
                print('Ok')
                

            

            t.sleep(0.5)
            subprocess.call(["./dio4test", '0', '0'])  
            #subprocess.call(["./dio4test", '0', '0']) 
            input_pint = subprocess.check_output(['./dio4test'])
            #print(input_pint)
            sta=str(input_pint)
            status=sta.split(' ')
            #print(status)
            return(Quality) 
            
                      

        except genicam.GenericException as e:
            # Error handling.
            print("An exception occurred.")
            print(e.GetDescription())
            exitCode = 1
    elif retry == 'High':
        #print('Part not detected')
        pass
        
    else:
        return(grabimage(retry))
                
subprocess.call(["./dio4test", '0', '0']) 
input_pint = subprocess.check_output(['./dio4test'])
#print(input_pint)
sta=str(input_pint)
status=sta.split(' ')
#print(status)

# Number of images to be grabbed.
countOfImagesToGrab = 1

# The exit code of the sample application.
exitCode = 0
maxCamerasToUse = 3



while status[3] == 'Low':
    #subprocess.call(["./dio4test", '0', '0']) 
    input_pint = subprocess.check_output(['./dio4test'])
    #print(input_pint)
    sta=str(input_pint)
    status=sta.split(' ')
    #print(status)
    grabimage(status[2])
        
        
print('Start Conveyor')
sys.exit(exitCode)
