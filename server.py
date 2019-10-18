## @package Сервер flask
# принимает данные (изображения) с клиента и находит на них
# контуры, которые похожи на контуры заданного изображения
from flask import Flask, request, Response, render_template
import numpy as np
import cv2
import threading
import time

# путь к заданному изображению
URL_IMG = "/home/ozik/Workspace/flask_stream_prod/images/fig_crop.jpg"
# переменная, которая содержит в себе изображение
outputFrame = None
# отдельный поток
lock = threading.Lock()

app = Flask(__name__)

def images_conversion(img, i_crop):
    image = img.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray_c = cv2.GaussianBlur(i_crop, (3, 3), 0)
    edged = cv2.Canny(gray, 70, 250)
    edged_c = cv2.Canny(gray_c, 70, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    kernel_c = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_c = cv2.morphologyEx(edged_c, cv2.MORPH_CLOSE, kernel_c)

    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts_c = cv2.findContours(closed_c.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    return cnts, cnts_c

def img_matching(img, i_crop, span=0.028):
    cnts, cnts_c = images_conversion(img, i_crop)
    huhu_c = []
    for cnt_c in cnts_c:
        peri = cv2.arcLength(cnt_c, True)
        approx = cv2.approxPolyDP(cnt_c, 0.001 * peri, True)
        m = cv2.moments(approx)
        hus = cv2.HuMoments(m)
        huhu_c.append(hus)
        cv2.drawContours(i_crop, [approx], -1, (0, 255, 0), 2)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.001 * peri, True)
        m = cv2.moments(approx)
        hus = cv2.HuMoments(m)
        for h in enumerate(huhu_c):
            if -span<(max(h[1])-max(hus))<span:
                cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)
    return img

def img_matching_alternative(img, i_crop, comp_factor=0.1):
    cnts, cnts_c = images_conversion(img, i_crop)
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.001 * peri, True)
        for cnt_c in cnts_c:
            peri_c = cv2.arcLength(cnt_c, True)
            approx_c = cv2.approxPolyDP(cnt_c, 0.001 * peri_c, True)
            sd = cv2.matchShapes(approx, approx_c, 1, 0)
            if sd < comp_factor:
                cv2.drawContours(img, [approx], -1, (0, 0, 145), 2)
    return img

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
def index():
    global outputFrame
    if request.method == 'POST':
        r = request
        nparr = np.fromstring(r.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        outputFrame = image.copy()
        return render_template('index.html')
    else:
        return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)