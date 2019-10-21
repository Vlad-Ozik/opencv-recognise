## @package server
#
# It receives data (images) from the client and finds on them 
# contours similar to the contours of a given image
from flask import Flask, request, Response
import numpy as np
import cv2
import threading
import time
from config import URL_IMG, HOST, PORT


## A variable that contains an image
outputFrame = None
## thread
lock = threading.Lock()
app = Flask(__name__)

## @brief Function to find the contours of uploaded images
# @param img image sent to the server 
# @param i_crop image whose contour will be searched
# @return cnts, cnts_crop - contours of img and i_crop images respectively
#
# To find the contours, the following actions are performed:
# - make img b/w, i_crop is already b/w
# - blur image
# - detect edges of objects in the image
# - make a morphological transformation
# - find the contour
def find_contours(img, i_crop):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray_crop = cv2.GaussianBlur(i_crop, (3, 3), 0)
    edged = cv2.Canny(gray, 70, 250)
    edged_crop = cv2.Canny(gray_crop, 70, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    kernel_crop = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_crop = cv2.morphologyEx(edged_crop, cv2.MORPH_CLOSE, kernel_crop)

    cnts = cv2.findContours(closed.copy(), 
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts_crop = cv2.findContours(closed_crop.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

    return cnts, cnts_crop

## @brief Function for finding similar contours
# @param img image that was sent to the server
# @param i_crop image whose contour will be searched
# @param span everything from -span to span is a similar outline
# @return img final image with a similar outline highlighted, if any
#
# To find similar contours, the following actions are performed:
# - find the contours of images using the find_contours function
# - in the loop find Hu moments of the i_crop image 
# contours and enter them into an array
# - in a loop, find the xy moments of the contours of the image img and 
# compare them with the xy moments of the contours of the image i_crop, 
# if the difference between them is in the range -span and span, then the 
# contour is applied to img
#
# By default, span = 0.028. To find similar contours from a poor quality 
# web camera,this value is the best (it was obtained experimentally). 
# If the images are of good quality, the span parameter can be reduced 
# to reduce the risk of finding dissimilar contours.
def img_matching(img, i_crop, span=0.028):
    cnts, cnts_crop = find_contours(img, i_crop)
    huhu_c = []
    for cnt_crop in cnts_crop:
        peri = cv2.arcLength(cnt_crop, True)
        approx = cv2.approxPolyDP(cnt_crop, 0.001 * peri, True)
        moment = cv2.moments(approx)
        hus = cv2.HuMoments(moment)
        huhu_c.append(hus)

    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.001 * peri, True)
        moment = cv2.moments(approx)
        hus = cv2.HuMoments(moment)
        for h in enumerate(huhu_c):
            if -span<(max(h[1])-max(hus))<span:
                cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)
    return img

## @brief Alternative function for finding similar contours
# @param img image that was sent to the server
# @param i_crop image whose contour will be searched
# @param comp_factor maximum contour similarity indicator
# @return img final image with a similar outline highlighted, if any
#
# To find similar contours, the following actions are performed:
# - find the contours of images using the find_contours function
# - in the loop, the contours are compared using the matchShapes function, 
# which returns an indicator of the similarity of the contours
#
# By default, comp_factor=0.1. To find similar contours from a poor quality 
# web camera,this value is the best (it was obtained experimentally). 
# If the images are of good quality, the comp_factor parameter can be reduced 
# to reduce the risk of finding dissimilar contours.
def img_matching_alternative(img, i_crop, comp_factor=0.1):
    cnts, cnts_crop = find_contours(img, i_crop)
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.001 * peri, True)
        for cnt_crop in cnts_crop:
            peri_c = cv2.arcLength(cnt_crop, True)
            approx_c = cv2.approxPolyDP(cnt_crop, 0.001 * peri_c, True)
            mshapes = cv2.matchShapes(approx, approx_c, 1, 0)
            if mshapes < comp_factor:
                cv2.drawContours(img, [approx], -1, (0, 0, 145), 2)
    return img

## @brief Function generator
# @return string string with an inserted image for output in HTML format 
#
# In a function, an image is read in a separate stream, 
# the contour of which will be searched, 
# after which one of the two functions is used to search for similar contours. 
# The resulting final image is converted to jpg format and sent 
def generate():
    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue
            i_crop = cv2.imread(URL_IMG, 0)
            img = img_matching_alternative(outputFrame, i_crop)
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            if not flag:
                continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
 			bytearray(encodedImage) + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
## @brief Function accepting requests from the client application
# @return string  
#
# The function accepts a POST and GET request. 
# If GET, then sends the user a webcast; if POST, 
# then it reads the image and is entered into the global variable
def index():
    global outputFrame
    if request.method == 'POST':
        req = request
        nparr = np.fromstring(req.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        outputFrame = image.copy()
        return ('', 204)
    else:
        return Response(generate(), \
                        mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(HOST, PORT)