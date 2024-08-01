import json
import requests
from PIL import Image
from io import BytesIO
import os

# Função para recortar e salvar os objetos em pastas separadas por classe
def crop_and_save_objects(json_data, output_folder):
    # Certifica-se de que a pasta de saída existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Itera sobre os projetos no JSON
    for project_id, project_info in json_data['projects'].items():
        for label in project_info['labels']:
            for annotation in label['annotations']['objects']:
                bbox = annotation['bounding_box']
                class_name = annotation['name']
                image_url = json_data['data_row']['row_data']

                # Cria a pasta para a classe, se ainda não existir
                class_folder = os.path.join(output_folder, class_name)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

                # Faz o download da imagem
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))

                # Calcula as coordenadas do recorte
                left = bbox['left']
                top = bbox['top']
                right = left + bbox['width']
                bottom = top + bbox['height']

                # Recorta a imagem
                cropped_image = image.crop((left, top, right, bottom))

                # Salva a imagem recortada na pasta da classe
                # Usando feature_id para nome único
                feature_id = annotation['feature_id']
                cropped_image.save(os.path.join(class_folder, f'{feature_id}.jpg'))

# Exemplo de uso
json_file_path = '/home/joaopdrm/projetos_pessoais/retail-store-product-detection/teste.ndjson'
output_folder = '/home/joaopdrm/projetos_pessoais/retail-store-product-detection/data'

# Lê o arquivo JSON
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Chama a função para processar o JSON e criar o dataset
crop_and_save_objects(data, output_folder)

