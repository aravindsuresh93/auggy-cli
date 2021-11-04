from utils.load_annotations import LoadAnnotations
from utils.annotations.save_annotations.save_factory import SaveAnnotations

loader =LoadAnnotations()
saver = SaveAnnotations.get(".txt")

images_folder = "test/yolo"
annotations_folder = "test/yolo"
artefacts_path = "test/yolo/artefacts"

df, classes = loader.load(images_folder, annotations_folder, artefacts_path)
print(df, classes)



# from utils.augmentor import  Augment
# aug = Augment()

