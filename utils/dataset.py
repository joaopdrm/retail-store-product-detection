import json
import requests
from PIL import Image
from io import BytesIO
import os
import random
from sklearn.model_selection import train_test_split

def crop_and_save_objects(json_data, output_folder, split_ratio=0.8):
    """
    Esta função cria um dataset a partir de um json de objetos rotulados na plataforma
    labelbox e divide o dataset em treino e validação.
    """
    # Certifica-se de que as pastas de saída existam
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # Lista para armazenar os caminhos das imagens e os dados de anotação
    image_paths = []
    annotations = []

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
                feature_id = annotation['feature_id']
                file_path = os.path.join(class_folder, f'{feature_id}.jpg')
                cropped_image.save(file_path)

                # Adiciona o caminho da imagem e as anotações à lista
                image_paths.append(file_path)
                annotations.append((file_path, class_name))

    # Divide os dados em treino e validação
    train_paths, val_paths = train_test_split(image_paths, test_size=1 - split_ratio, random_state=42)

    # Cria um dicionário para rastrear as pastas da classe
    class_folders = {class_name: set() for _, class_name in annotations}

    # Move os arquivos para as pastas de treino e validação
    for file_path in train_paths:
        class_name = annotations[image_paths.index(file_path)][1]
        dest_folder = os.path.join(train_folder, class_name)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        dest_path = os.path.join(dest_folder, os.path.basename(file_path))
        os.rename(file_path, dest_path)
        class_folders[class_name].add(dest_path)
    
    for file_path in val_paths:
        class_name = annotations[image_paths.index(file_path)][1]
        dest_folder = os.path.join(val_folder, class_name)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        dest_path = os.path.join(dest_folder, os.path.basename(file_path))
        os.rename(file_path, dest_path)
        class_folders[class_name].add(dest_path)

    # Remove pastas de classe vazias
    for class_name, paths in class_folders.items():
        if not paths:
            for folder in [train_folder, val_folder]:
                class_folder = os.path.join(folder, class_name)
                if os.path.exists(class_folder):
                    os.rmdir(class_folder)

def process_json_file(json_file_path, output_folder, split_ratio=0.8):
    # Lê o arquivo JSON
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Itera sobre as imagens no JSON
    if 'images' in data:
        for image_data in data['images']:
            crop_and_save_objects(image_data, output_folder, split_ratio)
    else:
        # Se o JSON não contém uma lista de imagens, processa como antes
        crop_and_save_objects(data, output_folder, split_ratio)

# Exemplo de uso
json_file_path = 'caminho/para/seu/json' #deve estar junto das imagens
output_folder = 'caminh/para/pasta/de/output'

# Chama a função para processar o JSON, criar o dataset e dividir em treino e validação
process_json_file(json_file_path, output_folder)

