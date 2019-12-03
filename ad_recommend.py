# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:28:31 2019

@author: lpc
"""
import numpy as np
import cv2
from Video import Video
import random
import operator
import time
def ad_broadcast(video):
    url = video.url
    cap = cv2.VideoCapture(url)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    loop_flag = 0
    result = True
    while(cap.isOpened()):
        ret,frame = cap.read()
        cv2.imshow(video.label,frame)
        loop_flag = loop_flag + 1
        if cv2.waitKey(1) & loop_flag == frames:
            break
        if cv2.waitKey(40) & 0xFF == ord('q'):
            result = False
            break
        if cv2.waitKey(40) & 0xFF == ord('s'):
            cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    return result
def play_Video(file_names):
    cnt = 0
    result = True
    j = 0
    while j < 100:
        j = j + 1
        if cnt < len(file_names):
            #result = ad_broadcast(file_names[cnt])
            label_num = file_names[cnt].label_num
            file_names[cnt].count = file_names[cnt].count +1
            file_names[cnt].label_count[label_num] = file_names[cnt].label_count[label_num] + 1
            print("播放....."+str(file_names[cnt].ad_id)+"类别："+file_names[cnt].label+" 类别播放次数" +str(file_names[cnt].label_count[label_num])+" 播放次数:" +str(file_names[cnt].count))

        #查找下一个播放的类别
        label = random.randint(0,11)
        print("下一个视频的lable="+str(label))        
        #查找对应类别要播放的视频
        broad_list = []
        for i in range(len(file_names)):
            if file_names[i].label_num == label:
                broad_list.append(file_names[i])
        cmpfun = operator.attrgetter('count','ad_id')
        broad_list.sort(key = cmpfun)
        if len(broad_list) >0:
            cnt = broad_list[0].ad_id
        else:
            cnt = cnt+1 if (cnt+1<len(file_names)) else 0
        if result == False:
            break
        
def read_Video():
    file_names = []
    path = "F:\\Desktop\\opencv_detection\\"
    for i in range(0,50):
        label = random.randint(0,11)
        video = Video(i,path+"video"+str(i%3+1)+".mp4","label"+str(label),label,0)
        file_names.append(video)
    return file_names

if __name__ =="__main__":
    file_names = read_Video()
    if len(file_names)>0:
        play_Video(file_names)
    else:
        print("无法找到视频资源！！！")
    file_names = sorted(file_names,key = lambda x:(x.label_num,x.ad_id))
    for file in  file_names:
        label_num = file.label_num
        print(str(file.ad_id) + " " + file.url   + " " + file.label  + " "+ str(label_num) + " " +str(file.label_count[label_num]) +" " +str(file.count))
        