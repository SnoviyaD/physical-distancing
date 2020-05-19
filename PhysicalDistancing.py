import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import sys
import math

class Detection:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
                

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])


class PhysicalDistancing:
    def __init__(self,object,img):
        self.threshold=0.7
        self.avgHeight = 165   #in cm
        self.centroids = []
        boxes, scores, classes, num = obj1.processFrame(img)
        self.pick = []
        self.img = img
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > self.threshold:
                box = boxes[i]
                self.pick.append(box)
                centroid = self.centroid(box[1], box[0], box[3], box[2])
                self.centroids.append(centroid)
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(0,255,0),2)

    #calculate centroid of rect
    def centroid(self,xA, yA, xB, yB):
        midpointX = (xA + xB)/2
        midpointY = (yA + yB)/2
        return [midpointX,midpointY]

    #calculate distance between two rects in pixels
    def distance(self,xA1, xA2, xB1, xB2, i, j):
        inf = sys.maxsize
        a = abs(xA1-xB2)
        b = abs(xA2-xB1)
        c = abs(self.centroids[i][0] - self.centroids[j][0])

        xDist = min(a if a>0 else inf, b if b>0 else inf, c)
        xDist = xDist**2
        yDist = abs(self.centroids[i][1] - self.centroids[j][1])**2
        sqDist = xDist + yDist
        return math.sqrt(sqDist)

    def checkDistancing(self):
        img = self.img
        for i in range(len(self.pick)-1):
            boxI = self.pick[i]
            (xA1, yA1, xB1, yB1) = (boxI[1], boxI[0], boxI[3], boxI[2])
            for j in range(i+1,len(self.pick)):
                boxJ = self.pick[j]
                (xA2, yA2, xB2, yB2) = (boxJ[1], boxJ[0], boxJ[3], boxJ[2])

                #calculate distance in pixels
                dist = self.distance(xA1, xA2, xB1, xB2, i, j)

                #calculate actual distance in cm
                heightI = abs(yA1 - yB1)
                heightJ = abs(yA2 - yB2)

                if heightI==0 or heightJ==0:
                    continue

                ratioI = self.avgHeight/heightI     # in cm/pixels
                ratioJ = self.avgHeight/heightJ

                meanRatio = (ratioI + ratioJ)/2

                dist = dist * meanRatio       # in cm
                

                if dist<100:
                    cv2.rectangle(img,(xA1,yA1),(xB1,yB1),(0,0,255),2)
                    cv2.rectangle(img,(xA2,yA2),(xB2,yB2),(0,0,255),2)


if __name__ == "__main__":
    #Edit your training model path
    model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'   
    obj1 = Detection(path_to_ckpt=model_path)
    
    #Edit your input video path
    cap = cv2.VideoCapture('TownCentreXVID.avi')                                

    while True:
        r, img = cap.read()

        #resize frame
        height, width, layers = img.shape
        new_h=height/2
        new_w=width/2
        img = cv2.resize(img, (int(new_w), int(new_h)))

        # Verify if physical distancing rules are followed or not
        obj2 = PhysicalDistancing(obj1, img)
        obj2.checkDistancing()

        #Display frame
        cv2.imshow("frame", img)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows() 
