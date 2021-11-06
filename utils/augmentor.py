import albumentations as A
import pandas as pd
import cv2
import os
import random



class Augment:
    def __init__(self):
        self.bbox_params = A.BboxParams(format='pascal_voc')

    """
    Pixel Level
    """

    def Blur(self, blur_limit):
        return A.Compose([A.Blur(blur_limit=(3, blur_limit), p=1)], bbox_params=self.bbox_params)

    def Clahe(self, clip_limit):
        return A.Compose([A.CLAHE(clip_limit=clip_limit, tile_grid_size=(8, 8), p=1)], bbox_params=self.bbox_params)

    def ChannelDropout(self, channel_drop_range, fill_value):
        return A.Compose([A.ChannelDropout(channel_drop_range=channel_drop_range,fill_value= fill_value, p=1)], bbox_params=self.bbox_params)

    def ChannelShuffle(self):
        return A.Compose([A.ChannelShuffle(p=1)], bbox_params=self.bbox_params)

    def ChannelShuffle(self, brightness, contrast, saturation, hue):
        return A.Compose([A.ChannelShuffle(brightness=brightness, contrast=contrast,saturation=saturation, hue=hue, p=1)], bbox_params=self.bbox_params)
    
    def Downscale(self, scale_min, scale_max):
        return A.Compose([A.Downscale(scale_min=scale_min, scale_max=scale_max,p=1)], bbox_params=self.bbox_params)

    def Equalize(self, mode, by_channels):
        return A.Compose([A.Equalize(mode=mode, by_channels=by_channels, p=1)], bbox_params=self.bbox_params)
    
    def FancyPCA(self, alpha):
        return A.Compose([A.FancyPCA(alpha = alpha)], bbox_params=self.bbox_params)   

    def GaussNoise(self, var_limit, mean, per_channel):
        return A.Compose([A.GaussNoise(var_limit = var_limit, mean = mean, p=1)], bbox_params=self.bbox_params)

    def GlassBlur(self, sigma, max_delta, iterations, mode, p=1):
        return A.Compose([A.GlassBlur(sigma = sigma, max_delta = max_delta, iterations=iterations, mode = mode, p=1)], bbox_params=self.bbox_params)        

    def GaussianBlur(self, blur_limit, sigma_limit):
        return A.Compose([A.GaussianBlur(blur_limit = blur_limit, sigma_limit = sigma_limit,p=1)], bbox_params=self.bbox_params)        
         

    """
    Workflow
    """
    def response(self, transformed):
        boxes = []
        labels = []
        for box in transformed["bboxes"]:
            boxes.append(box[:4])
            labels.append(box[-1])
        image = transformed['image']
        return image, boxes, labels

    def request(self, transform_data):
        self.transformation_type = transform_data.get("transformation_type", "")
        assert len(self.transformation_type), "Transformation type not available"

        self.transformation_parameters = transform_data.get("transformation_parameters", {})
        # assert len(self.transformation_parameters), "Transformation parameters not available"

    def get_arg(self, argname, default_val):
        return self.transformation_parameters.get(argname, default_val)


    def augment(self, image, boxes, transform_data):
        
        self.request(transform_data)

        switcher = {
            "Blur" : [self.Blur, (self.get_arg("blur_limit", 7),)],
            "Clahe" : [self.Clahe, (self.get_arg("clip_limit", 4),)],
            "ChannelDropout" : [self.ChannelDropout, (self.get_arg("channel_drop_range", (1, 1)), self.get_arg("fill_value", 0))],
            # "ChannelShuffle" : self.ChannelShuffle(),
            # "ColorJitter" : self.ColorJitter(self.get_arg("brightness", 0.2), self.get_arg("contrast", 0.2),self.get_arg("saturation", 0.2),self.get_arg("hue", 0.2)),
            "Downscale" : [self.Downscale, (self.get_arg("scale_min", 0.25), self.get_arg("scale_max", 0.25))],
            "Equalize" : [self.Equalize, (self.get_arg("mode", "cv"), self.get_arg("by_channels", True))],
            "FancyPCA" : [self.FancyPCA, (self.get_arg("alpha", 0.1),)],
            "GaussNoise" : [self.GaussNoise, (self.get_arg("var_limit", [10,50]), self.get_arg("mean", 0), self.get_arg("per_channel", True))],
            "GaussianBlur" : [self.GaussianBlur, (self.get_arg("blur_limit", (3, 7)), self.get_arg("sigma_limit",0))],
            "GlassBlur" : [self.GlassBlur, (self.get_arg("sigma", 0.7), self.get_arg("max_delta", 4), self.get_arg("iterations", 2), self.get_arg("mode", "fast"))]
        }


        if self.transformation_type in switcher.keys():
            func, args = switcher.get(self.transformation_type)
            transform =  func(*args) #init transform
            transformed = transform(image=image, bboxes=boxes)
            return self.response(transformed)


    def process(self, df, input_images_folder, output_images_folder, transformation_types = [], random_select = False):

        for image_name in df['image_name'].unique():
            name, _ = os.path.splitext(image_name)
            sdf = df[df['image_name'] == image_name]
            boxes = []
            img = cv2.imread(os.path.join(input_images_folder, image_name))
            for _, row in sdf.iterrows():
                xmin, ymin, xmax, ymax = row['box_xmin'], row['box_ymin'], row['box_xmax'], row['box_ymax']
                boxes.append([xmin, ymin, xmax, ymax, row['label']])


            for transformation_type in transformation_types:
                image, boxes, labels = self.augment(img, boxes, {'transformation_type' : transformation_type})

                #Image Save
                augmented_name = f'{name}_{transformation_type}'
                augmented_image_name = augmented_name + '.jpg'
                
                cv2.imwrite(os.path.join(output_images_folder, augmented_image_name), image)
                height, width, depth = image.shape

                for box, label in zip(boxes, labels):
                    data = {'label' : label, 
                            'box_xmin' : box[0],
                            'box_ymin' : box[1],
                            'box_xmax' : box[2],
                            'box_ymax' : box[3],
                            'box_h' : box[3] - box[1],
                            'box_w' : box[2] - box[0],
                            'image_name' : augmented_image_name,
                            'annotation_path' : "",
                            'image_width': width,
                            'image_height' :height,
                            'image_depth' :depth
                            }

                    sdf = pd.DataFrame(data, [0])
                    df = df.append(sdf)

            # Save/ update df code





"""
Blur - blur_limit (4-100)
CLAHE - clip_limit (1-100)
ChannelDropout - Randomly Drop Channels in the input Image. - channel_drop_range (1,2) , fill_value(0,255)
ChannelShuffle - Randomly rearrange channels of the input RGB image - no  params
ColorJitter - Randomly changes the brightness, contrast, and saturation of an image. brightness [0-1], contrast[0-1],saturation[0-1], hue[0-1]
Downscale - Decreases image quality by downscaling and upscaling back. - "scale_min" - [0-0.5], "scale_max" - [0-0.5]
Equalize - Equalize the image histogram. mode='cv'/'pil', by_channels=True
** FDA - Not available
FancyPCA -  
** FromFloat
GaussNoise
GaussianBlur
GlassBlur

HistogramMatching
HueSaturationValue
IAAAdditiveGaussianNoise
IAAEmboss
IAASharpen
IAASuperpixels
ISONoise
ImageCompression
InvertImg
MedianBlur
MotionBlur
MultiplicativeNoise
Normalize
Posterize
RGBShift
RandomBrightnessContrast
RandomFog
RandomGamma
RandomRain
RandomShadow
RandomSnow
RandomSunFlare
Sharpen
Solarize
ToFloat
ToGray
ToSepia
"""
