## @package server
#
# Принимает данные (изображения) с клиента и находит на них
# контуры, которые похожи на контуры заданного изображения
from flask import Flask, request, Response, render_template
import numpy as np
import cv2
import threading
import time

## путь к заданному изображению
URL_IMG = "/home/ozik/Workspace/flask_stream_prod/images/book.jpg"
## переменная, которая содержит в себе изображение
outputFrame = None
## отдельный поток
lock = threading.Lock()

app = Flask(__name__)

## @brief Функция для нахождения контуров переданных изображений
# @param img изображение, которое прислали на сервер 
# @param i_crop изображение, контур которого будет искатся
# @return cnts, cnts_c - контуры изображений img и i_crop соотвественно
#
# Для нахождения контуров производятся действия:
# - сделать img черно-белой, i_crop уже ч/б
# - размыть изображение
# - обнаружить края объектов на изображении
# - произвести морфологическую трансформацию
# - найти контур
def find_contours(img, i_crop):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

## @brief Функция для поиска похожих контуров
# @param img изображение, которое прислали на сервер 
# @param i_crop изображение, контур которого будет искатся
# @param span всё в диапазоне от -span до span является похожим контуром
# @return img конечное изображение с выделеным похожим контуром, если он есть
#
# Для нахождения похожих контуров производятся действия:
# - найти контуры изображений с помощью функции find_contours
# - в цикле найти Ху моменты контуров изображения i_crop и занести их в массив
# - в цикле  в цикле найти Ху моменты контуров изображения img и сравнить их 
# c Ху моментами контуров изображения i_crop, если разница между их ними
# находится между -span и span, то контур наноситься на img
#
# По умолчанию параметр span=0.028. Для нахождения похожих контуров с
# веб камеры плохого качества это значения является наилучшим (было полчено
# эксперементальным путём). Если изображения хорошего качества, параметр span 
# можно уменьшить, чтобы уменьшить риск нахождения не схожих контуров.  
def img_matching(img, i_crop, span=0.028):
    cnts, cnts_c = find_contours(img, i_crop)
    huhu_c = []
    for cnt_c in cnts_c:
        peri = cv2.arcLength(cnt_c, True)
        approx = cv2.approxPolyDP(cnt_c, 0.001 * peri, True)
        m = cv2.moments(approx)
        hus = cv2.HuMoments(m)
        huhu_c.append(hus)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.001 * peri, True)
        m = cv2.moments(approx)
        hus = cv2.HuMoments(m)
        for h in enumerate(huhu_c):
            if -span<(max(h[1])-max(hus))<span:
                cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)
    return img

## @brief Альтернативная функция для поиска похожих контуров
# @param img изображение, которое прислали на сервер 
# @param i_crop изображение, контур которого будет искатся
# @param comp_factor максимальный показатель схожести контуров
# @return img конечное изображение с выделеным похожим контуром, если он есть
#
# Для нахождения похожих контуров производятся действия:
# - найти контуры изображений с помощью функции find_contours
# - в цикле сравниваются сами контуры с помощью функции matchShapes,
# которая возвращяет показатель сходства контуров
#
# По умолчанию параметр comp_factor=0.1. Для нахождения похожих контуров с
# веб камеры плохого качества это значения является наилучшим (было полчено
# эксперементальным путём). Если изображения хорошего качества, параметр span 
# можно уменьшить, чтобы уменьшить риск нахождения не схожих контуров.
def img_matching_alternative(img, i_crop, comp_factor=0.1):
    cnts, cnts_c = find_contours(img, i_crop)
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

## @brief Функция генератор
# @return string строка с вставленным изображением для вывода в формате HTML 
#
# В функции, в отдельном потоке, считывается изображение, контур которого
# будет искатся, после применяется одина из двух функций для поиска похожих контуров.
# Полученное конечное изображение переводится в формат jpg и отправляется 
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
## @brief Функция принимающая запросы от клиентского приложения
# @return string строка  
#
# Функция принимает POST и GET запрос. Если GET, то отправляет пользователю 
# трансляцию; если POST, то считывает изображение и заносится в глобальную переменную
def index():
    global outputFrame
    if request.method == 'POST':
        r = request
        nparr = np.fromstring(r.data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        outputFrame = image.copy()
        return ('', 204)
    else:
        return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)