import dxcam
import torch
import cv2
import numpy as np
import time
import os
# MONITOR_WIDTH = 1920
# MONITOR_HEIGHT = 1080
# r1egion = (int(MONITOR_WIDTH/2-MONITOR_WIDTH/2),int(MONITOR_HEIGHT/2-MONITOR_HEIGHT/2),int(MONITOR_WIDTH/2+MONITOR_WIDTH/2),int(MONITOR_HEIGHT/2+MONITOR_HEIGHT/2))
# x,y,width,height = region


left, top = (1920 - 415), (1080 - 415)
right, bottom = left + 415, top + 415
region = (left, top, right, bottom)

model = torch.hub.load(r"C:\Users\metal\Desktop\VALO\yolov5", "custom", path=r"C:\Users\metal\Desktop\VALO\415x415.pt", source="local")
print(region)
model.apm = True
model.conf = 0.4
camera = dxcam.create(device_idx=0, output_idx=0, output_color="BGRA")
#region NP STUFF
enemy_redside = np.array([[1814.0259740259742, 871.948051948052], [1776.6233766233768, 906.2337662337662], [1714.2857142857144, 859.4805194805195], [1754.8051948051948, 845.4545454545455],[1800,845.4545454545455], [1814.0259740259742, 871.948051948052]], np.int32)
enemy_blueside = np.array([[1849.8701298701299,901.5584415584416],[1812.4675324675325,937.4025974025974],[1868.5714285714287,991.948051948052],[1849.8701298701299,901.5584415584416]], np.int32)
ally_blueside = np.array([[1689.3506493506495,887.5324675324675],[1751.6883116883118,932.7272727272727],[1712.7272727272727,973.2467532467533],[1692.4675324675325,965.4545454545455], [1689.3506493506495,887.5324675324675]], np.int32)
ally_redside = np.array([[1779.7402597402597,963.8961038961039],[1835.844155844156,1020],[1757.922077922078,1029.3506493506493],[1740.7792207792209,1005.9740259740261], [1779.7402597402597,963.8961038961039]], np.int32)

red_side_top = np.array([[1682.3176823176811,843.3566433566428],[1699.100899100898,857.7422577422572],[1725.4745254745244,840.55944055944],[1814.9850149850138,842.1578421578416],[1814.185814185813,825.3746253746249],[1705.4945054945044,827.3726273726269]], np.int32)
top_side_river = np.array([[1774.6253746253735,927.672327672327],[1761.0389610389598,916.4835164835159],[1753.4465534465523,906.8931068931063],[1738.6613386613376,903.2967032967027],[1725.0749250749238,892.1078921078915],[1717.4825174825164,879.7202797202791],[1713.0869130869119,866.9330669330664],[1703.8961038961027,857.7422577422572],[1695.104895104894,868.1318681318676],[1705.0949050949039,877.7222777222771],[1709.4905094905084,894.1058941058935],[1719.4805194805183,905.6943056943051],[1732.6673326673315,912.4875124875119],[1749.8501498501487,919.2807192807187],[1762.237762237761,924.8751248751242],[1766.6333666333655,939.660339660339]], np.int32)

blue_side_top = np.array([[1679.120879120878,846.9530469530464],[1693.106893106892,859.3406593406588],[1682.3176823176811,875.7242757242751],[1682.7172827172817,983.2167832167826],[1666.3336663336652,982.8171828171821],[1664.7352647352636,876.5234765234759],[1668.3316683316673,860.9390609390604]], np.int32)
blue_side_bot = np.array([[1864.5354645354632,1019.9800199800193],[1884.9150849150838,1043.9560439560432],[1845.3546453546442,1055.1448551448545],[1741.8581418581407,1055.1448551448545],[1741.0589410589398,1039.560439560439],[1838.9610389610377,1038.761238761238]], np.int32)

bot_side_river = np.array([[1795.8041958041947,946.4535464535459],[1812.987012987012,961.638361638361],[1828.5714285714273,967.632367632367],[1842.1578421578408,975.624375624375],[1854.9450549450537,992.0079920079913],[1859.7402597402586,1006.7932067932061],[1870.5294705294693,1013.9860139860133],[1858.1418581418568,1023.5764235764229],[1845.7542457542445,1011.588411588411],[1847.352647352646,1000.7992007992001],[1839.360639360638,988.8111888111881],[1825.7742257742245,981.2187812187806],[1808.191808191807,974.0259740259734],[1787.0129870129858,959.2407592407586]], np.int32)
red_side_bot = np.array([[1886.113886113885,1036.3636363636356],[1868.5314685314672,1017.9820179820173],[1879.7202797202785,1005.1948051948045],[1878.9210789210777,907.2927072927067],[1896.5034965034952,907.692307692307],[1898.101898101897,1013.1868131868125]], np.int32)

baron_rift = np.array([[1727.8721278721268,875.3246753246748],[1741.8581418581407,872.1278721278716],[1755.8441558441548,887.3126873126868],[1749.8501498501487,901.6983016983011],[1731.06893106893,893.3066933066928],[1723.8761238761228,881.3186813186808]], np.int32)
dragon_pit = np.array([[1814.185814185813,983.2167832167826],[1812.5874125874113,995.2047952047946],[1822.5774225774214,1005.9940059940053],[1838.5614385614374,1004.3956043956038],[1842.1578421578408,997.2027972027965],[1837.7622377622365,982.4175824175818],[1826.9730269730258,978.0219780219774]], np.int32)

red_side_mid = np.array([[1777.0229770229757,932.0679320679315],[1791.8081918081907,943.656343656343],[1844.155844155843,889.7102897102891],[1832.5674325674313,882.1178821178815]], np.int32)
blue_side_mid = np.array([[1774.6253746253735,936.8631368631362],[1789.4105894105883,948.4515484515479],[1732.6673326673315,1003.596403596403],[1716.6833166833155,988.4115884115878]], np.int32)
start_time = time.time()
x = 1
counter = 0

def Check_Inside(name:str, insideval:int, region:str):
    if name == "Enemy_Minimap" and insideval == 1.0:
        print("Enemy Inside "+ region)

lower_red = np.array([100, 20, 20])
upper_red = np.array([255, 100, 100])
#endregion

def get_icons(icon_size, champions):

    icons = []
    folder = os.listdir(r"C:\Users\metal\Desktop\VALO\icons")
    for i in champions:
        if i in folder:    
            champion = i[:i.find('.')]
            extension = i[i.rfind('.'):]
            if extension != '.png':
                continue

        p = os.path.join(r"C:\Users\metal\Desktop\VALO\icons", i)
        if not os.path.isfile(p):
            continue
        img = cv2.imread(p)

        img = cv2.resize(img, icon_size)

        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        icons.append((champion, img))

    return icons
template_img = cv2.imread(r"C:\Users\metal\Desktop\VALO\icons\veigar.png")

champs = ["veigar.png", "teemo.png", "ashe.png", "rumble.png", "evelynn.png", "pyke.png", "kaisa.png", "senna.png"]
icons = get_icons((35,35), champs)
while True:  
    try:
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
            xmin = int(df.iloc[i,0])
            ymin = int(df.iloc[i,1]) 
            xmax = int(df.iloc[i,2])
            ymax = int(df.iloc[i,3])
            cv2.putText(screenshot, df.loc[i,"name"], (xmin,ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,  
                    0.5, (255,255,255), 1, cv2.LINE_AA) 
            
            rect = cv2.rectangle(screenshot,(xmin,ymin), (xmax, ymax), (255, 255,255), 1)

            img = screenshot[xmin:(xmin+xmax), ymin:(ymin+ymax)]
            
            img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            top_champ = ""
            val = 0
            for champion, icon in icons:
                res = cv2.matchTemplate(img_2, icon, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.7)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                                 
                if max_val > 0.5:
                    cv2.putText(screenshot, champion, (xmax, ymax + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    except:
         print("", end="")
             



    
    cv2.imshow("ColorShit", screenshot)
    if(cv2.waitKey(1) == ord('q')):
        cv2.destroyAllWindows()
        break
    