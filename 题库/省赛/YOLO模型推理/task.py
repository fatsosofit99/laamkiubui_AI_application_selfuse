import json
import cv2
from cv2.dnn import NMSBoxes
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

class ObjectDetector:
    def __init__(self, model_path: str, label_path: str, nms_threshold: float = 0.5, score_threshold: float = 0.5):
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        #TODO
        self.session=ort.InferenceSession(model_path)
        with open(label_path,'r')as f:
            self.labels = json.load(f)
    def preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        #TODO
        img=cv2.imread(image_path)
        original_img=cv2.resize(img,(640,640),cv2.INTER_LINEAR)
        input_img=cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
        input_img=input_img.astype(np.float32)/255.0 
        input_img=input_img.transpose(2,0,1)
        input_img=np.expand_dims(input_img,0)
       
        return input_img,original_img
    def run_inference(self, image: np.ndarray) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[str]]:
        #TODO
        input_name=self.session.get_inputs()[0].name
        output_name=self.session.get_outputs()[0].name
        outputs=self.session.run([output_name],{input_name:image})
        preds = np.squeeze(outputs[0]).T
        boxes=[]
        scores=[]
        class_names=[]
        img_h,img_w=640,640
        for pred in preds:
            class_score=pred[4:]
            score =np.max(class_score.astype(np.float32))
            if score>self.score_threshold:
                class_id=np.argmax(class_score)

                xc,yc,w,h=pred[:4]
                x1=xc-w/2
                y1=yc-h/2
                x2=xc+w/2
                y2=yc+h/2

                boxes.append([float(x1),float(y1),float(x2),float(y2)])
                scores.append(float(score))
                class_names.append(self.labels[str(class_id)])
        return boxes,scores,class_names

    def apply_nms(self, boxes: List[Tuple[float, float, float, float]], scores: List[float], class_names: List[str]) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[str]]:
        #TODO
        nms_boxes=[]
        for box in boxes:
            x1,y1,x2,y2 = box
            nms_boxes.append([x1,y1,(x2-x1),(y2-y1)])
        indices=NMSBoxes(boxes,scores,self.score_threshold,self.nms_threshold)
        filtered_boxes=[]
        filtered_scores=[]
        filtered_class_names=[]
        if len(indices)>0:
            for i in indices.flatten():
                filtered_boxes.append(boxes[i])
                filtered_scores.append(scores[i])
                filtered_class_names.append(class_names[i])
                
        return filtered_boxes, filtered_scores, filtered_class_names

    def detect_objects(self, image_path: str) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]], List[float], List[str]]:
        image, original_image = self.preprocess(image_path)
        boxes, scores, class_names = self.run_inference(image)
        filtered_boxes, filtered_scores, filtered_class_ids = self.apply_nms(boxes, scores, class_names)
        return original_image, filtered_boxes, filtered_scores, filtered_class_ids

def draw_detections(image: np.ndarray, boxes: List[Tuple[float, float, float, float]], scores: List[float], class_names: List[str]) -> np.ndarray:
    for box, score, class_name in zip(boxes, scores, class_names):
        x1, y1, x2, y2 = map(int, box)
        label = f"Class {class_name}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    model_path = 'yolov8l.onnx'
    image_path = 'example.jpg'
    label_path = 'index_to_name.json'
    
    detector = ObjectDetector(model_path=model_path, label_path=label_path)
    original_image, boxes, scores, class_names = detector.detect_objects(image_path=image_path)
    
    result_image = draw_detections(original_image, boxes, scores, class_names)
    cv2.imwrite("detections.png", result_image)