import cv2
import os
#from openai import OpenAI
import tkinter as tk
import pydirectinput as pyautogui
import time
pyautogui.PAUSE=0
def drawPolygons(polygonList, image_to_draw_on):
    for i in polygonList:
        image_to_draw_on = cv2.polylines(image_to_draw_on, [i[0]], True, (255,255,255), 1)

def point_test(polygon, point, champ, alignment, time):
    res = cv2.pointPolygonTest(polygon[0], point, False)
    to_send = []
    if res == 1.0:
        to_send.append(alignment + "," + champ + "," + polygon[1])

        return to_send


def move_ping(x,y):
    current_mouse_pos = pyautogui.position()
    screen_x, screen_y = 1920, 1080
    pyautogui.moveTo(screen_x-(416 - x), screen_y-(416 - y))
    pyautogui.press("g")
    pyautogui.leftClick(duration=0.1)
    time.sleep(0.01)
    pyautogui.mouseUp()
    pyautogui.moveTo(current_mouse_pos[0], current_mouse_pos[1])



def get_icons(icon_size, champions):
    icons = []
    folder = os.listdir(r"champion")
    for i in champions:
        if i in folder:    
            champion = i[:i.find('.')]
            extension = i[i.rfind('.'):]
            if extension != '.png':
                continue

        p = os.path.join(r"champion", i)
        if not os.path.isfile(p):
            continue
        img = cv2.imread(p)
        img = cv2.resize(img, icon_size)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        icons.append((champion, img))
    return icons


