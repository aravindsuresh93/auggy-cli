from utils.annotations.open_annotations.open_factory import OpenAnnotations, OpenLabels
from utils.annotations.annotation_format.annotation_format import FormatFinder
from config.config import BASE_FOLDER, SUPPORTED_EXTENSIONS
import pandas as pd
import time
import os


from clogger.clogger import CLogger
logger = CLogger.get("auggy-load-annotations")



class LoadAnnotations:
    """
    Load annotationfiles and convert them into standardized dataframe
    """
    @staticmethod
    def combine_annotation_to_frame(annotation_info):
        df = pd.DataFrame()   
        for f, info in annotation_info.items():
            if len(info.get("error", "")):
                sdf = pd.DataFrame(info, [0])
                df = df.append(sdf) if len(df) else  sdf
            main_info = info.copy()
            if 'bounding_box' in main_info.keys(): main_info.pop('bounding_box')
            for bbox in info.get('bounding_box', []):
                combined = {**main_info, **bbox}
                sdf = pd.DataFrame(combined, [0])
                df = df.append(sdf) if len(df) else  sdf
        df['error'] = 0 if "error" not in df.columns else df['error'].fillna(0)
        df['deteled'], df['modified'], df['soft_error'] = False, False, False
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def convert_classes_to_frame(classes):
        classes = pd.DataFrame.from_dict(classes, orient = 'index', columns=['label'])
        classes['deleted'] = False
        return classes

    @staticmethod
    def load_files(image_folder, annotation_files = [], annotation_formats=[], artefacts_path=""):
        annotation_info = {}
        for annotation_format in annotation_formats:
            classes = OpenLabels.get(annotation_format)(artefacts_path).classes
            AE = OpenAnnotations.get(annotation_format)
            for file_name in annotation_files:
                name, ext = os.path.splitext(file_name)
                name = os.path.basename(name)
                if annotation_format not in ext: continue
                annotation = AE.open(file_name, image_folder, name, classes)
                annotation_info[file_name] = annotation
        df = LoadAnnotations.combine_annotation_to_frame(annotation_info)
        classes = LoadAnnotations.convert_classes_to_frame(classes)
        return df, classes
    
    @staticmethod
    def load(images_folder, annotations_folder, artefacts_path=""):
        annotation_files, annotation_formats = [], {}
        for file in os.listdir(annotations_folder):
            ext = os.path.splitext(file)[1]
            if ext in SUPPORTED_EXTENSIONS:
                annotation_files.append(os.path.join(annotations_folder, file))
                annotation_formats[ext] = 1

        annotation_formats = list(annotation_formats.keys())
        logger.info(f"Annotation formats {annotation_formats}")

        df, classes = LoadAnnotations.load_files(images_folder, annotation_files, annotation_formats,artefacts_path)
        return df, classes


        













