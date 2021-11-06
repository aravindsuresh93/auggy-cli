from utils.load_annotations import LoadAnnotations
from utils.annotations.save_annotations.save_factory import SaveAnnotations
from utils.augmentor import Augment


loader =LoadAnnotations()
saver = SaveAnnotations.get(".txt")
augmentor = Augment()

images_folder = "test/yolo"
annotations_folder = "test/yolo"
artefacts_path = "test/yolo/artefacts"
output = "test/out"

df, classes = loader.load(images_folder, annotations_folder, artefacts_path)








    










# saver.save(df, classes, output)

