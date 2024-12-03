import dxcam
import torch
import cv2
import numpy as np
import time
import os
#from openai import OpenAI
import random
from LeagueFuncs import *
from ctypes import windll
import pydirectinput as PDI
import traceback

left, top = (1920 - 415), (1080 - 415)
right, bottom = left + 415, top + 415
region = (left, top, right, bottom)

model = torch.hub.load("yolov5", "custom", path="415x415.pt", source="local")
print(region)
model.apm = True
model.conf = 0.6
camera = dxcam.create(device_idx=0, output_idx=0, output_color="BGRA")

#Polygons for Map regions
#region NP STUFF
blueside_blue = [np.array([[66.79310344827586,252.3793103448276],[126.86206896551724,271.2068965517241],[190.9655172413793,208.44827586206898],[162.72413793103448,177.9655172413793],[126.86206896551724,164.9655172413793],[108.93103448275862,143.89655172413794],[94.13793103448276,108.93103448275862],[68.58620689655173,109.82758620689656]], np.int32), "blueside_blue"]
blueside_red = [np.array([[147.48275862068965,289.1379310344828],[168.10344827586206,355.9310344827586],[313.7931034482759,351],[323.6551724137931,329.0344827586207],[307.9655172413793,315.58620689655174],[306.17241379310343,303.0344827586207],[308.41379310344826,299.44827586206895],[303.48275862068965,294.51724137931035],[290.9310344827586,307.0689655172414],[274.7931034482759,306.62068965517244],[262.2413793103448,300.3448275862069],[257.7586206896552,290.48275862068965],[256.86206896551727,282.86206896551727],[258.2068965517241,273.8965517241379],[260.8965517241379,267.62068965517244],[214.27586206896552,229.9655172413793]], np.int32), "blueside_red"]
redside_blue = [np.array([[233.55172413793105,209.79310344827587],[255.0689655172414,237.13793103448276],[296.7586206896552,254.6206896551724],[318.7241379310345,273.8965517241379],[331.7241379310345,306.17241379310343],[345.17241379310343,319.17241379310343],[355.48275862068965,306.17241379310343],[356.82758620689657,293.17241379310343],[357.2758620689655,163.6206896551724],[322.3103448275862,157.79310344827587],[298.1034482758621,147.48275862068965]], np.int32), "redside_blue"]
redside_red = [np.array([[257.7586206896552,63.6551724137931],[98.62068965517241,76.20689655172414],[99.51724137931035,86.51724137931035],[114.75862068965517,96.37931034482759],[120.13793103448276,111.17241379310344],[120.58620689655173,123.72413793103449],[142.55172413793105,110.27586206896552],[158.6896551724138,111.17241379310344],[165.41379310344828,120.13793103448276],[169,133.13793103448276],[168.10344827586206,147.93103448275863],[167.6551724137931,154.20689655172413],[209.79310344827587,192.75862068965517],[272.55172413793105,132.24137931034483]], np.int32), "redside_red"]
baronrift_pit = [np.array([[143.44827586206895,116.55172413793103],[128.6551724137931,129.55172413793105],[135.82758620689654,146.58620689655172],[159.58620689655172,150.17241379310346],[165.41379310344828,131.79310344827587],[158.24137931034483,116.10344827586206]], np.int32), "baronrift pit"]
dragon_pit = [np.array([[266.2758620689655,268.9655172413793],[260,284.6551724137931],[271.2068965517241,301.6896551724138],[292.2758620689655,298.55172413793105],[294.0689655172414,282.41379310344826],[284.2068965517241,267.62068965517244]], np.int32), "dragon pit"]
redside_top = [np.array([[257.3103448275862,39.44827586206897],[100.41379310344828,42.13793103448276],[79.3448275862069,52.44827586206897],[63.6551724137931,71.27586206896552],[90.55172413793103,87.41379310344827],[112.51724137931035,69.0344827586207],[158.6896551724138,64.10344827586206],[260.44827586206895,63.6551724137931]], np.int32), "redside top"]
redside_bot = [np.array([[384.62068965517244,163.6206896551724],[383.7241379310345,298.55172413793105],[378.7931034482759,323.6551724137931],[370.2758620689655,339.7931034482759],[358.17241379310343,348.7586206896552],[336.6551724137931,327.2413793103448],[352.7931034482759,314.2413793103448],[358.17241379310343,302.1379310344828],[358.62068965517244,290.48275862068965],[357.7241379310345,163.17241379310346]], np.int32), "redside bot"]
redside_mid = [np.array([[203.51724137931035,200.3793103448276],[224.58620689655172,218.31034482758622],[299.44827586206895,142.10344827586206],[281.9655172413793,125.06896551724138]], np.int32), "redside mid"]
botside_river = [np.array([[98.62068965517241,85.62068965517241],[115.65517241379311,101.3103448275862],[120.13793103448276,115.65517241379311],[121.0344827586207,127.75862068965517],[135.82758620689654,142.10344827586206],[153.31034482758622,154.20689655172413],[177.51724137931035,170.79310344827587],[208,194.10344827586206],[191.86206896551724,207.10344827586206],[182.89655172413794,190.9655172413793],[158.6896551724138,178.86206896551724],[131.79310344827587,167.20689655172413],[112.96551724137932,150.6206896551724],[101.3103448275862,120.13793103448276],[82.48275862068965,107.13793103448276]], np.int32), "topside river"]
topside_river = [np.array([[234.89655172413794,215.17241379310346],[215.17241379310346,229.0689655172414],[249.6896551724138,255.51724137931035],[280.62068965517244,269.86206896551727],[307.51724137931035,294.0689655172414],[307.51724137931035,307.51724137931035],[310.2068965517241,318.2758620689655],[323.6551724137931,327.6896551724138],[341.58620689655174,315.1379310344828],[328.58620689655174,304.3793103448276],[326.3448275862069,291.3793103448276],[320.9655172413793,277.0344827586207],[312,266.2758620689655],[292.2758620689655,253.72413793103448],[274.3448275862069,247.89655172413794],[255.51724137931035,238.48275862068965]], np.int32), "botside river"]
blueside_top = [np.array([[39.89655172413793,253.27586206896552],[66.3448275862069,253.27586206896552],[66.3448275862069,121.48275862068965],[73.51724137931035,108.93103448275862],[84.72413793103449,104],[60.06896551724138,79.79310344827586],[49.758620689655174,85.17241379310344],[43.93103448275862,101.75862068965517],[40.3448275862069,115.65517241379311]], np.int32), "blueside top"]
blueside_bot = [np.array([[160.93103448275863,355.48275862068965],[160.48275862068965,380.58620689655174],[304.82758620689657,380.58620689655174],[327.2413793103448,376.1034482758621],[345.62068965517244,367.1379310344828],[355.0344827586207,357.2758620689655],[331.7241379310345,334.41379310344826],[322.3103448275862,339.3448275862069],[316.9310344827586,351.44827586206895],[287.7931034482759,359.9655172413793],[259.1034482758621,356.82758620689657]], np.int32), "blueside bot"]
blueside_mid = [np.array([[127.75862068965517,274.7931034482759],[148.3793103448276,290.9310344827586],[218.31034482758622,220.55172413793105],[198.58620689655172,203.9655172413793]], np.int32), "blueside mid"]

all_polygon = []

all_polygon.append(blueside_blue)
all_polygon.append(blueside_red)

all_polygon.append(redside_blue)
all_polygon.append(redside_red)

all_polygon.append(baronrift_pit)
all_polygon.append(dragon_pit)

all_polygon.append(redside_top)
all_polygon.append(redside_bot)

all_polygon.append(redside_mid)
all_polygon.append(botside_river)

all_polygon.append(topside_river)
all_polygon.append(blueside_top)

all_polygon.append(blueside_bot)
all_polygon.append(blueside_mid)
#endregion

start_time = time.time()
x = 1
counter = 0
champs = ["Veigar.png", "Vi.png", "Smolder.png"]
enemy_team = ["Vi", "Smolder"]
jungle = "Vi"
player = "Veigar"
ally_team = ["Veigar"]
PDI.PAUSE=0
icons = get_icons((35,35), champs)
gpt_send = []
new_start_time = time.time()
while True:  
    try:
        
        draw_rectangle_cords = []
        screenshot = camera.grab(region) 
        if screenshot is None: continue
        
        df = model(screenshot, size=415).pandas().xyxy[0]
        
        counter += 1
        if(time.time() - start_time) > x:
                fps = "fps: " + str(int(counter/(time.time() - start_time)))
                print(fps)
                counter = 0
                start_time = time.time()
        for i in range(0, 10):
            try:
                xmin = int(df.iloc[i,0])
                ymin = int(df.iloc[i,1]) 
                xmax = int(df.iloc[i,2])
                ymax = int(df.iloc[i,3])
            except:
                print("", end="")
            screenshot = np.array(screenshot)
            draw_rectangle_cords.append((xmin,ymin,xmax,ymax))
            img = cv2.cvtColor(screenshot[ymin:ymax,xmin:xmax], cv2.COLOR_BGR2BGRA)
            center = ((xmin+xmax)/2, (ymin+ymax)/2)
            threshold = 0.4
            for champ, icon in icons:
                #print(icon.shape, "-", img.shape)
                res = cv2.matchTemplate(img, icon, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                
                cv2.rectangle(screenshot, (xmin,ymin), (xmax, ymax), (255, 255,255))
                #cv2.putText(screenshot, "player " + champ, (xmax, ymax + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if champ in ally_team:
                    if max_val > threshold:
                        for i in all_polygon:
                            if point_test(i, center, champ, "ally", new_start_time) is not None and not point_test(i, center, champ, "ally", new_start_time) in gpt_send:
                                pass
                        cv2.rectangle(screenshot,(xmin,ymin), (xmax, ymax), (255, 255,255), 1)
                        cv2.putText(screenshot, "ally " + champ, (xmax, ymax + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if champ in enemy_team:
                    if max_val > threshold:
                        for i in all_polygon:
                            if point_test(i, center, champ, "enemy", new_start_time) is not None and not point_test(i, center, champ, "enemy", new_start_time) in gpt_send:
                                if champ == jungle:    
                                    move_ping(int((xmax+xmin)/2), int((ymax+ymin)/2))
                                    ##########UNCOMMENT FOR TEXT ALERT#############
                                    # PDI.keyDown("enter")
                                    # PDI.keyUp("enter")
                                    # time.sleep(0.1)
                                    # PDI.write(f"{jungle} {i[1]}")
                                    # print(f"{jungle} {i[1]}")
                                    # PDI.keyDown("enter")
                                    # PDI.keyUp("enter")
                                    ##################################################
                                    time.sleep(2)
                        cv2.rectangle(screenshot,(xmin,ymin), (xmax, ymax), (255, 255,255), 1)
                        cv2.putText(screenshot, "enemy " + champ, (xmax, ymax + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                if champ == player:
                    if max_val > threshold:
                        for i in all_polygon:
                            if point_test(i, center, champ, "player", new_start_time) is not None and not point_test(i, center, champ, "player", new_start_time) in gpt_send:
                                if champ == jungle:
                                    pass
                        #print("What?")

    except Exception:
        traceback.print_exc()
    
    
    cv2.imshow("frame", screenshot)
    if(cv2.waitKey(1) == ord('q')):
        cv2.destroyAllWindows()
        break

    
