from ultralytics import YOLO

def finetuning(model_path:str,data_path:str,output_path:str, epochs:int, batch_size:int):
    """
    model_path: str
        Caminho para o diretorio do modelo
    data_path: str
        Caminho para o dataset.yaml
    output_path: str
        Caminho para o diretório onde será salvo o modelo
    epochs: int
        Quantidade de épocas que o modelo será treinado
    batch_size: int
        Tamanho do batch size
    """
    model = YOLO(model_path)
    model.train(data=data_path, epochs=epochs, batch_size=batch_size, name='fine-tuned-model')
    model.save(output_path)
    print(f"Modelo fine-tuned salvo em {output_path}")