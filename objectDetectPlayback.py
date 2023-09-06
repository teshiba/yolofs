import cv2
import Yolov8Lib as yolo
import cv2 as cv
import PySimpleGUI as sg
import sys

#####################################################################
# settings
#####################################################################
modelFile = 'yolov8n.pt'
poseMode = False

#####################################################################
# GUI settings
#####################################################################
font = cv.FONT_HERSHEY_SIMPLEX

layout = [[sg.Text('', key='TEXT')],
          [sg.Button('読み込み', key='BUTTON1')],
          [sg.Button('停止', key='BUTTON2', disabled=True)],
          [sg.Image(key='IMAGE', size=(400, 400))]]

window = sg.Window('v_cap_test', layout)

#####################################################################
# arguments
#####################################################################
args = sys.argv
path = args[1]

#####################################################################
# init vars
#####################################################################
switch = False
Yolov8Lib = yolo.Yolov8Lib(modelFile, poseMode)

#####################################################################
# main
#####################################################################
while True:
    event, values = window.read(timeout=0)

    if event in (sg.WIN_CLOSED, 'Exit'):
        if switch:
            vc.release()
        break

    if event == 'BUTTON1':
        vc = cv.VideoCapture(path)
        switch = vc.isOpened()
        window['BUTTON2'].update(disabled=False)
        window['BUTTON1'].update(disabled=True)

    if event == 'BUTTON2':
        vc.release()
        switch = False
        window['BUTTON2'].update(disabled=True)
        window['BUTTON1'].update(disabled=False)

    if switch:
        ret, flame = vc.read()
        if ret:
            img = Yolov8Lib.GenerateAnnotatedImage(flame)

            # if show another window
            # cv2.imshow("play video with yolov8 model names",origin)
            window['IMAGE'].update(cv.imencode('.png', img)[1].tobytes())

            cv2.waitKey(1)
        else:
            print('Read error EOF or fail to read frame.')
            window.close()
#####################################################################

