from utils.load_annotations import LoadAnnotations
from utils.annotations.save_annotations.save_factory import SaveAnnotations

loader =LoadAnnotations()
saver = SaveAnnotations.get(".txt")

images_folder = "test/yolo"
annotations_folder = "test/yolo"
artefacts_path = "test/yolo/artefacts"
output = "test/out"

df, classes = loader.load(images_folder, annotations_folder, artefacts_path)
# print(df, classes)
df['box_xmin'] = 0
saver.save(df, classes, output)





# from utils.augmentor import  Augment
# aug = Augment()

