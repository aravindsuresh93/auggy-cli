from utils.annotations.open_annotations.open_yolo import convert_to_yolo
import os

class SaveTXT:
    @staticmethod
    def convert_coordinates(size, box, classid):
        dw = 1.0/size[0]
        dh = 1.0/size[1]
        x = (box[0]+box[1])/2.0
        y = (box[2]+box[3])/2.0
        w = box[1]-box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y=y*dh
        h=h*dh
        return f'{classid} {x} {y} {w} {h}'

    # @staticmethod       
    # def save(info, classes, save_folder):
    #     yolo_text = ""
    #     for b in info:
    #         yololine = convert_to_yolo((b['width'], b['height']), (b['xmin'], b['xmax'], b['ymin'], b['ymax']), b['label'])
    #         yolo_text = yolo_text + yololine + "\n"
    #         txtfilename = os.path.basename(b['annotation_path'])

    #     with open(os.path.join(save_folder, txtfilename), 'w') as f:
    #         f.write(yolo_text)

    @staticmethod
    def save_classes(classes, save_folder):
        classes_txt = ""
        classes_array = classes[classes['deleted'] != False]['label'].values
        for c in classes_array:
            classes_txt = classes_txt + str(c) + "\n"
        with open(os.path.join(save_folder, "classes.txt"), 'w') as f:
            f.write(classes_txt)

    @staticmethod
    def save(df, classes, save_folder):
        for image_name in df['image_name'].unique():
            name, _ = os.path.splitext(image_name)
            sdf = df[df['image_name'] == image_name]
            yolo_text = ""
            for _, row in sdf.iterrows():
                x,y,w,h = convert_to_yolo(row['image_width'], row['image_height'], row['box_xmin'], row['box_ymin'], row['box_xmax'], row['box_ymax'])
                yololine = f"{row['label']} {x} {y} {w} {h}"
                yolo_text = yolo_text + yololine + "\n"

            with open(os.path.join(save_folder, f'{name}.txt'), 'w') as f:
                 f.write(yolo_text)
                 print('saved', f'{name}.txt')

