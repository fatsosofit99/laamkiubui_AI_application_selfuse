class ONNXModelInference:
    def __init__(self, 
                 model_path: str, 
                 input_size: Tuple[int, int] = (224, 224), 
                 mean: Tuple[float, float, float] = (0, 0, 0), 
                 scale: float = 1.0/255, 
                 swapRB: bool = True, 
                 crop: bool = False,
                 class_file: Optional[str] = None):

        self.model_path = model_path
        self.input_size = input_size
        self.mean = mean
        self.scale = scale
        self.swapRB = swapRB
        self.crop = crop
        self.classes: List[str] = []


        self.net = cv2.dnn.readNetFromONNX(model_path)
        if class_file is not None:
            with open(class_file, "r", encoding="utf-8") as f:
                self.classes = [line.strip() for line in f if line.strip()]
        #TODO
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=self.scale,
            size=self.input_size,
            mean=self.mean,
            swapRB=self.swapRB,
            crop=self.crop
        )
        return blob
        #TODO
    def infer(self, image: str) -> np.ndarray:
        
        img = cv2.imread(image)
        blob = self.preprocess(img)
        self.net.setInput(blob)
        output = self.net.forward()
        return output
        #TODO
    def postprocess(self, output: np.ndarray) -> Tuple[str, float]:
        
        scores = output.flatten()
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])

        if self.classes and 0 <= class_id < len(self.classes):
            label = self.classes[class_id]
        else:
            label = str(class_id)

        return label, score
        #TODO
if __name__ == "__main__":

    model_inference = ONNXModelInference(
        model_path="/home/project/model.onnx",
        input_size=(227, 227),
        mean=(123.68, 116.779, 103.939),
        scale=1.0,
        swapRB=True,
        crop=True,
        class_file="imagenet_classes.txt"
    )
    image = '/home/project/example.jpg'
    raw_output = model_inference.infer(image)
    label, score = model_inference.postprocess(raw_output)

    print(f"Predicted: {label} with score {score:.2f}")