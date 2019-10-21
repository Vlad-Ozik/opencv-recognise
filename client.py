## @package client
#
# Reads a stream from the camera, and sends it to the server
import requests
import json
import cv2
from config import ADDR, URL_IMG


## META data for sending message
content_type = 'image/jpeg'
headers = {'content-type': content_type}

## Link to the stream object 
cap = cv2.VideoCapture(0)

while True:
    flag, img_l = cap.read()
    img = cv2.flip(img_l, 1)
    _, img_encoded = cv2.imencode('.jpg', img)
    try:
        response = requests.post(ADDR, data=img_encoded.tostring(), \
                                 headers=headers)
    except:
        print('No access')
    cv2.imshow('stream', img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()


