# from utils.augmentor import  Augment
from utils.load_annotations import LoadAnnotations


loader =LoadAnnotations()
# aug = Augment()


images_folder = "test/yolo"
annotations_folder = "test/yolo"
artefacts_path = "test/yolo/artefacts"

df, classes = loader.load(images_folder, annotations_folder, artefacts_path)
print(df, classes)