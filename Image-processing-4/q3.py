import cv2
import numpy


def intersection_over_union(box1, box2):
    # 교집합 부분의 top left 좌표와 bottom right 좌표를 계산합니다.
    x1 = numpy.maximum(box1[0], box2[0])
    y1 = numpy.maximum(box1[1], box2[1])
    x2 = numpy.minimum(box1[2], box2[2])
    y2 = numpy.minimum(box1[3], box2[3])

    # 교집합의 넒이를 구합니다.
    intersection = numpy.maximum(x2 - x1, 0) * numpy.maximum(y2 - y1, 0)

    # 박스1의 넓이와 박스2의 넓이를 각각 구합니다.
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 두 박스의 넒이를 더한뒤 교집합 영역 넓이를 뺴면 합영역이 됩니다.
    union = box1_area + box2_area - intersection

    # iou를 계산합니다.
    iou = intersection / union
    return iou


def non_max_suppression(boxes, iou_threshold, confidence_threshold):
    nms_boxes = None
    
    # 박스의 confidence 값들을 내림차순으로 정렬하세요.
    boxes = boxes[numpy.where(boxes[:, 0] > confidence_threshold)]
    indices = numpy.argsort(-boxes, axis=0)[:, 0]
    
    # IoU를 계산하여 박스를 제거하세요.
    for i in range(boxes.shape[0]):
        for j in range(i + 1, boxes.shape[0]):
            if boxes[j, 0] == -1:
                continue
            
            iou = intersection_over_union(boxes[indices[i], 1:], boxes[indices[j], 1:])
            if iou >= iou_threshold:
                boxes[j, 0] = -1

    nms_boxes = boxes[numpy.where(boxes[:, 0] != -1)]   
   
    # 제거된 박스를 반환하세요.
    
    return nms_boxes


if __name__ == "__main__":
    box1 = [0.8, 100, 100, 170, 180]
    box2 = [0.9, 130, 140, 250, 300]
    box3 = [0.6, 100, 150, 290, 170]
    box4 = [0.7, 120, 170, 160, 190]
    box5 = [0.5, 110, 110, 290, 290]
    box6 = [0.3, 240, 200, 340, 270]

    boxes = numpy.array([
        box1, box2, box3, box4, box5, box6
    ])
    # 
    nms_boxes = non_max_suppression(boxes, 0.5, 0.3)
    nms_boxes = list(nms_boxes)
    print("제거된 박스 개수:",len(nms_boxes))  # 4
