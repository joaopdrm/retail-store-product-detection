import json
import cv2
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import mlflow

class ObjectDetectionEvaluator:
    def __init__(self, ndjson_path, image_path, model_path):
        self.ndjson_path = ndjson_path
        self.image_path = image_path
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.ground_truths = self.load_ground_truth()

    def load_ndjson(self):
        with open(self.ndjson_path, 'r') as f:
            data = [json.loads(line) for line in f]
        return data

    def load_ground_truth(self):
        data = self.load_ndjson()
        ground_truths = []
        for obj in data[0]['projects'].values():
            for label in obj['labels']:
                for annotation in label['annotations']['objects']:
                    box = annotation['bounding_box']
                    ground_truths.append({
                        'class': annotation['name'],
                        'bbox': [box['left'], box['top'], box['left'] + box['width'], box['top'] + box['height']]
                    })
        return ground_truths

    def get_model_predictions(self):
        image = cv2.imread(self.image_path)
        results = self.model(image)
        predictions = []
        for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
            predictions.append({
                'class': self.model.names[int(cls)],
                'bbox': box.cpu().numpy().astype(int).tolist()
            })
        return predictions

    def calculate_metrics(self, ground_truths, predictions, iou_threshold=0.5):
        gt_bboxes = np.array([gt['bbox'] for gt in ground_truths])
        pred_bboxes = np.array([pred['bbox'] for pred in predictions])
        pred_classes = [pred['class'] for pred in predictions]
        
        y_true = []
        y_pred = []
        
        for gt in ground_truths:
            gt_bbox = np.array(gt['bbox'])
            gt_class = gt['class']
            
            best_iou = 0
            best_pred = None
            
            for pred in predictions:
                pred_bbox = np.array(pred['bbox'])
                pred_class = pred['class']
                
                iou = self.compute_iou(gt_bbox, pred_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_class
            
            y_true.append(gt_class)
            if best_iou > iou_threshold:
                y_pred.append(best_pred)
            else:
                y_pred.append(None)
        
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        return precision, recall, f1

    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        
        iou = intersection / float(box1_area + box2_area - intersection)
        return iou

    def evaluate(self):
        # Iniciar uma nova execução no MLflow
        with mlflow.start_run():
            # Obter as predições do modelo
            predictions = self.get_model_predictions()
            
            # Calcular as métricas
            precision, recall, f1 = self.calculate_metrics(self.ground_truths, predictions)
            
            # Registrar as métricas no MLflow
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Exibir as métricas
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-Score: {f1:.2f}")

# Caminhos para os arquivos
ndjson_path = "path_to_your_ndjson_file.ndjson"
image_path = "path_to_your_image/your_image.jpg"
model_path = "path_to_your_model/last.pt"

# Criar uma instância da classe e avaliar o modelo
evaluator = ObjectDetectionEvaluator(ndjson_path, image_path, model_path)
evaluator.evaluate()

