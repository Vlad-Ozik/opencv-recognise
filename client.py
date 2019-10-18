## @package client
#
# Считывает поток с камеры, и отправляет на сервер
import requests
import json
import cv2

## Адрес сервера
addr = 'http://0.0.0.0:5000/'

## META данные для отправки сообщения
content_type = 'image/jpeg'
headers = {'content-type': content_type}

## Ссылка на объект стрима 
cap = cv2.VideoCapture(0)

while True:
    flag, img_l = cap.read()
    img = cv2.flip(img_l, 1)
    _, img_encoded = cv2.imencode('.jpg', img)
    try:
        response = requests.post(addr, data=img_encoded.tostring(), \
                    headers=headers)
    except:
        print('No access')
    cv2.imshow('stream', img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()


