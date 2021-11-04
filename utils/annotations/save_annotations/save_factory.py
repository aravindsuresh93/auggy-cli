class SaveAnnotations:
    @staticmethod
    def get(format):
        if format == ".xml":
            from utils.annotations.save_annotations.save_pascal import SaveXML
            return SaveXML()
        elif format == ".txt":
            from utils.annotations.save_annotations.save_yolo import SaveTXT
            return SaveTXT()
        else:
            raise NotImplementedError("Annotation Format not supported")