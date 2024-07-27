from ultralytics import YOLO

def train(epochs:int,image_size=int):
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="SKU-110K.yaml", epochs=epochs, imgsz=image_size)  # train the model

    return results