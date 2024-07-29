import json
import os
import shutil

def montar_dataset(json_path, images_base_dir, output_dir):
    # Diretórios para salvar as imagens e os arquivos de anotação
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')

    # Criar diretórios de saída, se não existirem
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Carregar o JSON
    try:
        with open(json_path, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Erro: O arquivo {json_path} não foi encontrado.")
        return
    except json.JSONDecodeError:
        print("Erro: Não foi possível decodificar o JSON.")
        return

    # Dicionário para armazenar as anotações por imagem
    annotations_dict = {}

    # Coletar anotações
    for item in data:
        try:
            image_name = item['data_row']['row_data']
            if image_name not in annotations_dict:
                annotations_dict[image_name] = []
            annotations_dict[image_name].append(item.get('annotations', []))
        except KeyError as e:
            print(f"Erro: Falta a chave {e} no item de dados.")
        except Exception as e:
            print(f"Erro inesperado: {e}")

    # Iterar sobre todos os arquivos de imagem na pasta base
    for image_name in os.listdir(images_base_dir):
        if image_name.endswith(('.jpg', '.png')):
            try:
                image_path = os.path.join(images_base_dir, image_name)

                # Copiar a imagem para o diretório de saída
                shutil.copy(image_path, os.path.join(images_dir, image_name))

                # Verificar se há anotações para a imagem
                annotations = annotations_dict.get(image_name, [])
                label_file_path = os.path.join(labels_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

                with open(label_file_path, 'w') as label_file:
                    for annotation_list in annotations:
                        for annotation in annotation_list:
                            bbox = annotation.get('bbox', {})
                            # Verificar se todas as chaves necessárias estão presentes
                            if all(k in bbox for k in ['left', 'top', 'width', 'height']):
                                x_center = (bbox['left'] + bbox['width'] / 2) / annotation.get('image_width', 1)
                                y_center = (bbox['top'] + bbox['height'] / 2) / annotation.get('image_height', 1)
                                width = bbox['width'] / annotation.get('image_width', 1)
                                height = bbox['height'] / annotation.get('image_height', 1)
                                class_id = annotation.get('class_id', 0)  # Valor padrão 0

                                # Escrever no arquivo de anotação
                                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                            else:
                                print(f"Warning: BBox está incompleto para a anotação {annotation}")

            except Exception as e:
                print(f"Erro inesperado ao processar {image_name}: {e}")

    print("Dataset montado com sucesso!")
