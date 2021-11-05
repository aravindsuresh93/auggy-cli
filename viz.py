import cv2
import os
from utils.annotations.open_annotations.open_yolo import BoundingBox, OpenTextFile
from utils.annotations.open_annotations.open_factory import OpenLabels

images_folder = "test/yolo"
annotations_folder = "test/out"
artefacts_path = "test/yolo/artefacts"
output = "test/out"

otf = OpenTextFile()


classes = OpenLabels.get('.txt')(artefacts_path).classes
print(classes)
for file_name in os.listdir(annotations_folder):
    if '.txt' not in file_name:
        continue
    name, ext = os.path.splitext(file_name)
    name = os.path.basename(name)

    img = cv2.imread(os.path.join(images_folder, f'{name}.jpg'))
    
    annotation = otf.open(os.path.join(annotations_folder ,file_name), images_folder, name, classes)
    bounding_box = annotation['bounding_box']
    
    
    for box in bounding_box:
        class_name = classes[box['label']]
        img = cv2.rectangle(img, (box['box_xmin'], box['box_ymin']), (box['box_xmax'], box['box_ymax']), (0,255,0), 1)
        cv2.putText(img, class_name, (box['box_xmin'], box['box_ymin']), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    cv2.imshow("sample",img)
    cv2.waitKey(0)

cv2.destroyAllWindows()