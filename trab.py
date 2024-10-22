import os  # operating system functionalities
from pathlib import Path  # for handling paths conveniently
import scipy.io  # load the data according to the instructions in the data sheet
import numpy as np  # images are numpy.ndarrays
import matplotlib.pyplot as plt  # for plotting the images

# Definir nome do diretório
path_input_dir = Path(
    "C:/Users/marce/OneDrive/Documentos/Faculdade/6Periodo/PAI/trabalhoPAI"
)
path_data = path_input_dir / "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"

# Carregar dados
data = scipy.io.loadmat(path_data)

# Acessar as imagens de 'data'
data_array = data["data"]
images = data_array["images"]

# Acesso à imagem m do paciente n (n varia de 0 a 54, m de 0 a 9)
n = 1  # Paciente 1
m = 5  # Imagem 5

imagem = images[0][n][m]
print(imagem.shape)  # Verifica o formato da imagem

# Plotar a imagem
plt.figure(figsize=(9, 9))
plt.imshow(imagem, cmap="gray")  # Exibir a imagem em escala de cinza
plt.axis("off")  # Esconder os eixos para melhor visualização
plt.show()
