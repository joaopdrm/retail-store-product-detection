import cv2
from ultralytics import YOLO

# Passo 1: Carregar o modelo YOLOv8 Shelf Object Detection
model = YOLO("path_to_your_model/last.pt")

# Passo 2: Configurar parâmetros (opcionais)
model.conf = 0.25  # Confiança mínima para detecção
model.iou = 0.45   # Limite de IoU para NMS
model.agnostic_nms = True  # NMS class-agnostic
model.max_det = 1000  # Número máximo de detecções por imagem

# Passo 3: Carregar a imagem da prateleira (substitua com o caminho da sua imagem)
image_path = "path_to_your_image/your_image.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Erro ao carregar a imagem. Verifique o caminho do arquivo.")
    exit()

# Passo 4: Executar detecção de objetos na imagem
results = model(image)

# Passo 5: Desenhar bounding boxes na imagem original
annotated_image = image.copy()

if isinstance(results, list):  # Verifica se 'results' é uma lista
    # Access bounding boxes and other information through the 'boxes' attribute
    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls
else:
    boxes = results.boxes.xyxy
    confidences = results.boxes.conf
    class_ids = results.boxes.cls

# Dicionário para armazenar a contagem de cada classe
class_counts = {name: 0 for name in model.names.values()}

# Iterar sobre as bounding boxes e desenhá-las na imagem
for box, conf, cls in zip(boxes, confidences, class_ids):
    xmin, ymin, xmax, ymax = box  # Obtém coordenadas
    class_name = model.names[int(cls)]
    class_counts[class_name] += 1  # Incrementa a contagem da classe correspondente

    cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)  # Desenha retângulo verde
    cv2.putText(annotated_image, f"{class_name} {conf:.2f}", (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Adiciona nome da classe e confiança

# Contar a quantidade de bounding boxes
num_boxes = len(boxes)
print(f"Quantidade de produtos na imagem: {num_boxes}")

# Adicionar contagem por classe na imagem
y_offset = 50
for class_name, count in class_counts.items():
    text = f"{class_name}: {count}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(annotated_image, (10, y_offset), (10 + text_width + 20, y_offset + text_height + 20), (255, 255, 255), -1)  # Desenha um retângulo branco com padding
    cv2.putText(annotated_image, text, (20, y_offset + text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Adiciona texto preto
    y_offset += text_height + 30

# Adicionar retângulo branco e número de bounding boxes na imagem
overall_text = f"Quantidade de produtos: {num_boxes}"
(text_width, text_height), baseline = cv2.getTextSize(overall_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
cv2.rectangle(annotated_image, (10, 10), (10 + text_width + 20, 10 + text_height + 20), (255, 255, 255), -1)  # Desenha um retângulo branco com padding
cv2.putText(annotated_image, overall_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Adiciona texto preto

# Passo 6: Mostrar a imagem com as bounding boxes
cv2.imshow("Annotated Image", annotated_image)  # Mostra a imagem com as bounding boxes

# Salvar a imagem com as bounding boxes desenhadas
cv2.imwrite("annotated_image.jpg", annotated_image)

# Fechar a janela de imagem com tecla 'q'
cv2.waitKey(0)
cv2.destroyAllWindows()
