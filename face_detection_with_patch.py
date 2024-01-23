
import numpy as np
import time
from facenet_pytorch import MTCNN
import torch

import mmcv, cv2


from PIL import Image, ImageDraw
from IPython import display

#ffpyplayer for playing audio
# from ffpyplayer.player import MediaPlayer

if torch.cuda.is_available():
    device = 'cuda:0'
    print("[INFO] Current device:", torch.cuda.get_device_name(torch.cuda.current_device()), f", device num:{torch.cuda.current_device()}")
elif torch.has_mps:
    device = 'mps'
    print("[INFO] Current device: MAC OS Metal Performance Shaders.")
else:
    device = 'cpu'
    print("[INFO] Current device: CPU")
    print("*"*50,"\n[WARNING] You may need to change the device to GPU or MPS to get better performance.")
    print("*"*50)
    
device = torch.device(device)
mtcnn = MTCNN(keep_all=True, device=device)

def shift_array(array, place):
    new_arr = np.roll(array, place, axis=1)
    new_arr[:,:place] = array[:,-place:]
    return new_arr

thresh = 25

# lower1 = np.array([114,153,145])   #(B,G,R)
# upper1 = np.array([157,212,200])  #

# lower2 = np.array([92,107,173])   #(B,G,R)
# upper2 = np.array([117,135,219])  #


lower1 = np.array([70,128,130])   #(B,G,R)
upper1 = np.array([150,206,210])  #

lower2 = np.array([88,80,194])   #(B,G,R)
upper2 = np.array([168,160,255])  #


# cap = cv2.VideoCapture('brian001.MOV')
cap = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mode = 'nblock'


if not cap.isOpened():
    print("Cannot open camera")
    exit()
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )   # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

# width = 360
# height = 540

fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('take3_out.avi',fourcc, fps, (int(width),int(height)))  #用來輸出影片
pri_count = 0




while True:
    start = time.time()
    ret, frame = cap.read()
    # audio_frame, val = player.get_frame()
    #frame = cv2.resize(frame,(width,height))
    faces, _ = mtcnn.detect(frame)
    # frame_draw = frame.copy()
    # draw = ImageDraw.Draw(frame)
    # frame_array = np.array(frame_draw)
    if not ret:
        print("Fin")
        break
    

                  # 縮小尺寸，避免尺寸過大導致效能不好
    points = []

    matrix1 = cv2.inRange(frame, lower1, upper1)
    matrix2 = cv2.inRange(frame, lower2, upper2)

    matrix1 = shift_array(matrix1,10)

    matrix = np.logical_and(matrix1,matrix2)

    tabel = np.where(matrix == True)

    for i in range(len(tabel[0])):
        points.append([tabel[0][i],tabel[1][i]])

    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # 將鏡頭影像轉換成灰階
    # faces = face_cascade.detectMultiScale(gray)      # 偵測人臉
    d = []
    if faces is not None:
        for (x, y, w, h) in faces:
            x = max(0, x)
            y = max(0, y)
            w = min(width,w)
            h = min(height,h)
            
            x = int(x)  
            y = int(y) 
            w = int(w)
            h = int(h)
            
            w = w - x
            h = h - y
            
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if w <= 1 or h <= 1:
                continue
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)    # 利用 for 迴圈，抓取每個人臉屬性，繪製方框
            mosaic = frame[y:y+h, x:x+w]   # 馬賽克區域
            level = 10                   # 馬賽克程度
            mh = int(h/level)            # 根據馬賽克程度縮小的高度
            mw = int(w/level)            # 根據馬賽克程度縮小的寬度
            if mw <= 0:              
                mw = 1  # 最小縮小比例為 1
            if mh <= 0:
                mh = 1  # 最小縮小比例為 1
            mosaic = cv2.resize(mosaic, (mw,mh), interpolation=cv2.INTER_LINEAR) # 先縮小
            mosaic = cv2.resize(mosaic, (w,h), interpolation=cv2.INTER_NEAREST)  # 然後放大
            center = [ (int(y + h / 2 + h * 1.2 )) , int(x + w / 2)]
            count = 0
            d.append(center)
            frame[y:y+h, x:x+w] = mosaic
            # for i in points:
            #     if np.sqrt((center[0] - i[0])**2 + (center[1] - i[1])**2) < np.sqrt(w*h)/2:
            #         count += 1
            # if mode == 'block':
            #     if count >= thresh:
            #         frame[y:y+h, x:x+w] = mosaic   # 將指定區域換成馬賽克區域
            # elif mode == 'nblock':
            #     if count <= thresh:
            #         frame[y:y+h, x:x+w] = mosaic   # 將指定區域換成馬賽克區域
            # print(count)
    out.write(frame)
    
    # 標示FPS
    end = time.time()
    cv2.putText(frame, f"FPS: {str(int(1 / (end - start)))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    start = end   
    cv2.imshow('face_mosaic', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord('1'):
        mode = 'block'
        print('mode 1')
    elif cv2.waitKey(1) == ord('2'):
        mode = 'nblock'
        print('mode 2')
    elif cv2.waitKey(1) == ord('3'):
        cv2.imwrite('output' + str(pri_count) + '.jpg', frame)
        pri_count += 1
        print('take a photo')
    # print('Time for a frame' + str(time.time() - start))
    




cap.release()
video.release()
out.release()
cv2.destroyAllWindows()
