import requests
import json
import cv2

addr = 'http://0.0.0.0:5000'
test_url = addr + '/'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

cap = cv2.VideoCapture(0)
#img = cv2.imread('/home/ozik/Workspace/flask_stream/block.jpg')

while True:
    flag, img_l = cap.read()
    img = cv2.flip(img_l, 1)
    _, img_encoded = cv2.imencode('.jpg', img)
    try:
        response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    except:
        print('dont ccess')
    cv2.imshow('ss', img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()


