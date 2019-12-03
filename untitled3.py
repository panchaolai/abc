import cv2 as cv
import time
from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

# 检测人脸并绘制人脸bounding box
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]  # 高就是矩阵有多少行
    frameWidth = frameOpencvDnn.shape[1]  # 宽就是矩阵有多少列
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    #  blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]]) -> retval  返回值   # swapRB是交换第一个和最后一个通道   返回按NCHW尺寸顺序排列的4 Mat值
    net.setInput(blob)
    detections = net.forward()  # 网络进行前向传播，检测人脸
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            x = int(x1)
            y = int(y1)
            w = int(x2-x1)
            h = int(y2-y1)
            if x < 0:
                w = w + x
                x =0
            if y < 0:
                h = h + y
                y = 0
            bboxes.append([x,y,w,h])  # bounding box 的坐标
            #cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)),
                #         8)  # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
    return frameOpencvDnn, bboxes

def face_recongnition():
    writeVideo_flag = False
    # 网络模型  和  预训练模型
    faceProto = "F:\\Desktop\\models\\opencv_face_detector.pbtxt"
    faceModel = "F:\\Desktop\\models\\opencv_face_detector_uint8.pb"
    
    ageProto = "F:\\Desktop\\models\\age_deploy.prototxt"
    ageModel = "F:\\Desktop\\models\\age_net.caffemodel"
    
    genderProto = "F:\\Desktop\\models\\gender_deploy.prototxt"
    genderModel = "F:\\Desktop\\models\\gender_net.caffemodel"
    # 模型均值
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    
    # 加载网络
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    # 人脸检测的网络和模型
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    
    # 打开一个视频文件或一张图片或一个摄像头
    cap = cv.VideoCapture(0)
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(cap.get(3))
        h = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		#创建视频流写入对象，VideoWriter_fourcc为视频编码器，15为帧播放速度。
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    padding = 20
    nms_max_overlap = 1.0
    max_cosine_distance = 0.3
    nn_budget = None
    model_filename = 'mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size = 1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    fps = 0.0
    
    while True:
        # Read frame
        hasFrame, frame = cap.read()
        t = time.time()
        frame = cv.flip(frame, 1)
        if not hasFrame:
            break
    
        frameFace, bboxs = getFaceBox(faceNet, frame)
        #print("box_num",len(bboxs))
        #print(bboxs)
        if not bboxs:
            print("No face Detected, Checking next frame")
            cv.imshow("Age Gender Demo", frameFace)
            continue
        features = encoder(frameFace,bboxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxs, features)]
        
        # Run non-maxima suppression.
        bboxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(bboxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
      
        #Call the tracker
        tracker.predict()
        tracker.update(detections)
    
    
          
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            
            #print(bbox)
            face = frame[max(0, int(bbox[1]) - int(padding)) : min(int(bbox[3]) + int(padding), frame.shape[0] - 1),
                         max(0, int(bbox[0]) - int(padding)) : min(int(bbox[2]) + int(padding), frame.shape[1] - 1)]
            print("=======", type(face), face.shape)  #  <class 'numpy.ndarray'> (166, 154, 3)
            #
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            #print("======", type(blob), blob.shape)  # <class 'numpy.ndarray'> (1, 3, 227, 227)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            #print("++++++", type(genderPreds), genderPreds.shape, genderPreds)   # <class 'numpy.ndarray'> (1, 2)  [[9.9999917e-01 8.6268375e-07]]  变化的值
            gender = genderList[genderPreds[0].argmax()]
            #print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
            
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            #print(agePreds[0].argmax())  # 3
            #print("*********", agePreds[0])   #  [4.5557402e-07 1.9009208e-06 2.8783199e-04 9.9841607e-01 1.5261240e-04 1.0924522e-03 1.3928890e-05 3.4708322e-05]
            #print("Age Output : {}".format(agePreds))
            #print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
            
            label = "{}, {}, {}".format(track.track_id , gender, age)
            cv2.rectangle(frameFace, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frameFace, label,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            if writeVideo_flag:
                #save a frame
                out.write(frame)
                frame_index = frame_index + 1
                list_file.write(str(frame_index)+' ')
                list_file.write(str(bbox[0]) + ' '+str(bbox[1]) + ' '+str(bbox[2]) + ' '+str(bbox[3]) + ' ')
                list_file.write('\n')
                        
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frameFace,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        cv.imshow("Age Gender Demo", frameFace)
        
 
        fps  = ( fps + (1./(time.time()-t)) ) / 2    
        print("fps=%f"%fps)            
        print("time : {:.3f} ms".format(time.time() - t))
        #Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    face_recongnition()
