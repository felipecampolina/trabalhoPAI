# Trabalho Prático - Diagnóstico de Esteatose Hepática em Exames de Ultrassom
# Integrantes(Matricula): Felipe Campolina(762732), Leandro Guido(777801) e Marcelo Augusto(775119)
import os
import csv
import re
import time
import random
import re
from typing import IO

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Button, Label, Menu, filedialog, simpledialog, messagebox, Toplevel, StringVar, Radiobutton, Entry, Frame, DoubleVar, IntVar
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops
import scipy.io
import pyfeats as pf
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# variaveis para facilitar debug
CARREGAR_DATASET_AUTOMATICO = False
LOG = True

class App(Frame):
    def __init__(self, root:Tk):
        super().__init__(root)

        self.root = root
        self.root.title("Visualizador de imagens em tons de cinza com ROIs")
        self.root.geometry("800x600")

        # criando barra de menu
        self.menu_bar = Menu(root)
        root.config(menu=self.menu_bar)

        # padrao SVM
        self.parametros_svm = {
            'kernel': 'linear',
            'C': 1.0,
            'gamma': 'scale',
            'degree': 3,
            'coef0': 0.0,
            'class_weight': None,
            'decision_function_shape': 'ovr'
        }

        # padrao MobileNet
        self.parametros_mobilenet = {
            'epochs': 5,
            'batch_size': 16,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'fine_tune_layers': 0,
            'early_stopping_patience': 3,
            'loss_function': 'binary_crossentropy',
            'dropout_rate': 0.0,
            'activation_function': 'relu',
            'momentum': 0.0
        }

        # menu arquivos
        self.menu_arquivos = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Arquivos", menu=self.menu_arquivos)
        self.menu_arquivos.add_command(label="Carregar banco de imagens (mat)", command=self.carregar_dataset_selecionavel)
        self.menu_arquivos.add_command(label="Carregar imagem (png ou jpg)", command=self.carregar_imagem_unica)
        self.menu_arquivos.add_command(label="Visualizar imagens", command=self.visualizar_imagens)

        # menu principal (opcoes)
        self.menu_opcoes = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Opções", menu=self.menu_opcoes)
        self.menu_opcoes.add_command(label="Carregar imagem para edição", command=self.carregar_imagem)


        self.menu_classificacao = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Classificação", menu=self.menu_classificacao)
        self.menu_classificacao.add_command(label="Classificar com SVM", command=self.classificar_com_svm)
        self.menu_classificacao.add_command(label="Classificar com MobileNet", command=self.classificar_com_mobilenet)
        self.menu_classificacao.add_command(label="Classificar e Comparar", command=self.classificar_e_comparar)
        self.menu_classificacao.add_command(label="Executar Modelo MobileNet Salvo", command=self.executar_modelo_salvo)
        self.menu_classificacao.add_command(label="Menu de Parâmetros", command=self.menu_de_parametros)
        # submenu para opcoes de ROI
        self.menu_roi = Menu(self.menu_opcoes, tearoff=0)
        self.menu_opcoes.add_cascade(label="Opções de ROI", menu=self.menu_roi)
        self.menu_roi.add_command(label="Gerar ROI tamanho livre", command=self.selecionar_roi)
        self.menu_roi.add_command(label="Mostrar histograma da ROI", command=self.mostrar_histograma_roi)
        self.menu_roi.add_command(label="Gerar ROIS 28x28(figado e rim)", command=self.gerar_rois_manual)
        self.menu_roi.add_command(label="Gerar ROIs 28x28 para CSV com descritores", command=self.gerar_rois_manual_dataset)
        self.menu_roi.add_command(label="Calcular GLCM e descritores de textura", command=self.calcular_textura_glcm)
        self.menu_roi.add_command(label="Calcular descritor de textura - SFM", command=self.calcular_textura_sfm)

        # opcao para mostrar hitograma da imagem completa
        self.menu_opcoes.add_command(label="Mostrar Histograma da Imagem", command=self.mostrar_histograma_da_img)

        # submenu de configuracoes
        self.settings_menu = Menu(self.menu_opcoes, tearoff=0)
        self.menu_opcoes.add_cascade(label="Configurações", menu=self.settings_menu)
        self.settings_menu.add_command(label="Ajustar escala X (Bins)", command=self.ajustar_bins)
        self.settings_menu.add_command(label="Ajustar escala Y (Limite)", command=self.ajustar_y_limit)
        self.settings_menu.add_command(label="Exibir métricas da imagem", command=self.mostrar_metricas_img)

        # --------------------------------------------------- VARIAVEIS


        # variaveis para configuracoes do histograma
        self.bins = 41       # padrão para o eixo X (bins)
        self.y_limit = 0.02  # padrão para o eixo Y

        # exibir imagem e ROI
        self.label = Label(root)
        self.label.pack()

        self.zoom_slider = None  # iniciando sem o slider
        self.img = None
        self.imagens = None
        self.total_patients = 0
        self.imagens_per_patient = 0
        self.current_patient = 0
        self.current_image = 0
        self.roi = None
        self.roi_zoom = None
        self.intensity_scale = 1
        self.zoom_factor = 1  # controle de zoom inicial
        
        # variaveis para o CSV
        self.csv_file : IO = None
        self.csv_writer : csv.DictWriter = None
        self.csv_path = "data.csv"  # nome fixo para o arquivo CSV

        # variavel para controlar a interrupecao do processo
        self.parar_processo = False

        if CARREGAR_DATASET_AUTOMATICO:
            self.carregar_dataset("dataset_liver_bmodes_steatosis_assessment_IJCARS.mat")

    # --------------------------------------------------- FUNCOES

    def carregar_dataset(self, path):
        if path:
            # carregar dados do arquivo .mat selecionado
            data = scipy.io.loadmat(path)
            data_array = data['data']
            self.imagens = data_array['images']
            self.total_patients = len(self.imagens[0])
            self.imagens_per_patient = len(self.imagens[0][0])
            messagebox.showinfo("SUCESSO", "Banco de imagens carregado com sucesso!")
            self.current_patient = 0
            self.current_image = 0
        else:
            messagebox.showwarning("AVISO", "Nenhum arquivo .mat foi selecionado.")

    def carregar_dataset_selecionavel(self):
        self.carregar_dataset(filedialog.askopenfilename(title="Selecione o arquivo .mat", filetypes=[("MAT files", "*.mat")]))

    def carregar_imagem_unica(self):
        path_img = filedialog.askopenfilename(title="Selecione a imagem", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if path_img:
            # carregar imagem
            self.img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
            if self.img is not None:
                self.mostrar_img()
            else:
                messagebox.showerror("ERRO", "Não foi possível carregar a imagem selecionada.")
        else:
            messagebox.showwarning("AVISO", "Nenhuma imagem foi selecionada.")

    def visualizar_imagens(self):
        if self.imagens is not None:
            janela_visualizar_imgs = Toplevel(self.root)
            janela_visualizar_imgs.title("Visualizar imagens")

            # label para exibir a imagem
            self.image_label = Label(janela_visualizar_imgs)
            self.image_label.pack()

            # label para exibir o num do pasciente e da foto
            self.info_label = Label(janela_visualizar_imgs, text="")
            self.info_label.pack()

            # botoes de navegacao
            prev_button = Button(janela_visualizar_imgs, text="Anterior", command=self.mostrar_imagem_ant)
            prev_button.pack(side='left')
            next_button = Button(janela_visualizar_imgs, text="Próxima", command=self.mostrar_prox_imagem)
            next_button.pack(side='right')

            # exibir a primeira imagem
            self.current_patient = 0
            self.current_image = 0
            self.mostrar_imagem_atual()
        else:
            messagebox.showwarning("AVISO", "Nenhum banco de imagens carregado. Carregue um banco de imagens primeiro.")

    def mostrar_imagem_atual(self):
        # carregar imagem do paciente e imagem atual
        self.img = self.imagens[0][self.current_patient][self.current_image]
        if self.img is not None:
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img))
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            # atualizar o texto com o numero do paciente e da foto
            info_text = f"Paciente: {self.current_patient}/{self.total_patients - 1}, Imagem: {self.current_image}/{self.imagens_per_patient - 1}"
            self.info_label.config(text=info_text)
        else:
            messagebox.showerror("ERRO", "Não foi possível carregar a imagem.")

    def mostrar_prox_imagem(self):
        self.current_image = (self.current_image + 1) % self.imagens_per_patient
        if self.current_image == 0:
            self.current_patient = (self.current_patient + 1) % self.total_patients
        self.mostrar_imagem_atual()

    def mostrar_imagem_ant(self):
        if self.current_image == 0:
            self.current_image = self.imagens_per_patient - 1
            self.current_patient = (self.current_patient - 1) % self.total_patients
        else:
            self.current_image -=1
        self.mostrar_imagem_atual()

    def carregar_imagem(self):
        if self.imagens is not None:
            # criar janela para escolher entre aletoria ou especifica
            choice_window = Toplevel(self.root)
            choice_window.title("Carregar imagem")
            choice_window.grab_set() # focar nessa janela

            escolha = StringVar(value="aleatoria")

            Label(choice_window, text="Escolha uma opção:").pack(pady=10, padx=80)

            Radiobutton(choice_window, text="Aleatória", variable=escolha, value="aleatoria").pack(anchor='center')
            Radiobutton(choice_window, text="Específica", variable=escolha, value="especifica").pack(anchor='center')

            def confirm_choice():
                choice = escolha.get()
                choice_window.destroy()
                if choice == 'aleatoria':
                    # carregar imagem aleatoria
                    self.current_patient = np.random.randint(0, self.total_patients)
                    self.current_image = np.random.randint(0, self.imagens_per_patient)
                    self.img = self.imagens[0][self.current_patient][self.current_image]
                    self.mostrar_img()
                elif choice == 'especifica':
                    self.perguntar_pela_imagem_e_paciente()
                else:
                    messagebox.showwarning("AVISO", "Opção inválida.")

            Button(choice_window, text="OK", command=confirm_choice).pack(pady=10)
        else:
            messagebox.showwarning("AVISO", "Nenhum banco de imagens carregado. Carregue um banco de imagens primeiro.")

    def perguntar_pela_imagem_e_paciente(self):
        input_window = Toplevel(self.root)
        input_window.title("Especificar imagem")
        input_window.grab_set() #  foca nesta janela

        Label(input_window, text=f"Digite o número do paciente (0-{self.total_patients - 1}):").pack(pady=5)
        patient_entry = Entry(input_window)
        patient_entry.pack(pady=5)

        Label(input_window, text=f"Digite o número da imagem (0-{self.imagens_per_patient - 1}):").pack(pady=5)
        image_entry = Entry(input_window)
        image_entry.pack(pady=5)

        def carregar_imagem_especifica():
            try:
                patient_num = int(patient_entry.get())
                image_num = int(image_entry.get())
                if 0 <= patient_num <= self.total_patients - 1 and 0 <= image_num <= self.imagens_per_patient - 1:
                    self.current_patient = patient_num
                    self.current_image = image_num
                    self.img = self.imagens[0][self.current_patient][self.current_image]
                    self.mostrar_img()
                    input_window.destroy()
                else:
                    messagebox.showerror("ERRO", "Números estao fora do intervalo válido.")
            except ValueError:
                messagebox.showerror("ERRO", "Insira números inteiros válidos.")

        Button(input_window, text="Carregar Imagem", command=carregar_imagem_especifica).pack(pady=10)

    def mostrar_img(self):
        print("mostrar_img")
        if self.img is not None:
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img))
            self.label.config(image=img_tk)
            self.label.image = img_tk
        else:
            messagebox.showerror("ERRO", "Não foi possivel carregar a imagem.")

    def selecionar_roi(self):
        if self.img is not None:
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)  # convercao para BGR para exibir com opencv
            self.roi = cv2.selectROI("Selecione a ROI", img_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Selecione a ROI")

            # cortar a roi da imagem original
            x, y, w, h = self.roi
            self.roi_zoom = self.img[y:y+h, x:x+w]

            # criar botao visualisar roi depois de selecionar
            if not hasattr(self, 'view_roi_button'):
                self.view_roi_button = Button(self.root, text="Visualizar ROI", command=self.mostrar_roi)
                self.view_roi_button.pack()
        else:
            messagebox.showwarning("AVISO", "Nenhuma imagem foi carregada. Carregue uma imagem primeiro.")

    def mostrar_roi(self):
        if self.roi_zoom is not None:
            plt.figure(figsize=(5, 5))
            plt.imshow(self.roi_zoom, cmap='gray')
            plt.title("ROI selecionada")
            plt.axis('on')
            plt.show()

    def mostrar_histograma_roi(self):
        if self.roi_zoom is not None:
            plt.figure()
            plt.hist(self.roi_zoom.ravel(), bins=self.bins, range=(0, 255), density=True, color='gray', edgecolor='black')
            plt.title("Histograma da ROI")
            plt.xlabel("Níveis de cinza")
            plt.ylabel("Densidade")
            plt.ylim(0, self.y_limit)
            plt.show()
        else:
            messagebox.showwarning("AVISO", "Nenhuma ROI de tamanho livre foi selecionada. Selecione uma ROI primeiro.")

    def calcular_textura_glcm(self):
        if self.roi_zoom is not None:  # se tem uma ROI selecionada
            entropias, homogeneidades, glcm_matrices = self.calcular_descritores_glcm(self.roi_zoom, exibir_matrizes=True)
            s = "Descritores de textura (GLCM) radial:\n\n"
            distancias = [1, 2, 4, 8]
            for i, d in enumerate(distancias):
                s += (f"Distância {d} pixels:\n"
                    f"Entropia: {entropias[i]:.4f}\n"
                    f"Homogeneidade: {homogeneidades[i]:.4f}\n\n")
            messagebox.showinfo("Descritores de textura (GLCM)", s)
        else:
            messagebox.showwarning("AVISO", "Nenhuma ROI foi selecionada. Selecione uma ROI primeiro.")


    def calcular_descritores_glcm(self, roi, exibir_matrizes=False):
        roi_norm = (roi / np.max(roi) * 255).astype(np.uint8)
        distancias = [1, 2, 4, 8]
        num_angulos = 16
        angulos = np.linspace(0, 2 * np.pi, num_angulos, endpoint=False)
    
        entropias = []
        homogeneidades = []
        glcm_matrices = []

        for idx, d in enumerate(distancias):
            glcm = graycomatrix(roi_norm, distances=[d], angles=angulos, levels=256, symmetric=True, normed=True)
            glcm_radial = np.sum(glcm, axis=3)
            glcm_radial = glcm_radial / np.sum(glcm_radial)

            glcm_matrices.append(glcm_radial)  # Armazenar a matriz GLCM

            glcm_nonzero = glcm_radial[glcm_radial > 0]
            entropia = -np.sum(glcm_nonzero * np.log2(glcm_nonzero))
            entropias.append(entropia)

            i_indices = np.arange(256).reshape(-1, 1)
            j_indices = np.arange(256).reshape(1, -1)
            homogeneidade = np.sum(glcm_radial / (1 + np.abs(i_indices - j_indices)))
            homogeneidades.append(homogeneidade)

            # Exibir a matris GLCM se solicitado
            if exibir_matrizes:
                self.mostrar_matriz_glcm(glcm_radial, d)

        return entropias, homogeneidades, glcm_matrices

    def mostrar_matriz_glcm(self, glcm_radial, distancia):
        plt.figure(figsize=(6, 5))
        plt.imshow(glcm_radial, cmap='gray')
        plt.colorbar()
        plt.title(f'Matriz GLCM Radial - Distância {distancia}')
        plt.xlabel('Níveis de cinza')
        plt.ylabel('Níveis de cinza')
        plt.show()

    def calcular_textura_sfm(self):
        if self.roi_zoom is not None:
            features, _ = pf.sfm_features(f=self.roi_zoom, mask=None, Lr=4, Lc=4)
            coarseness = features[0]
            contrast = features[1]
            periodicity = features[2]
            roughness = features[3]

            s = (f"Descritores de textura (SFM):\n\n"
                           f"Coarseness: {coarseness:.4f}\n"
                           f"Contrast: {contrast:.4f}\n"
                           f"Periodicity: {periodicity:.4f}\n"
                           f"Roughness: {roughness:.4f}")
            messagebox.showinfo("Descritores de textura (SFM)", s)
        else:
            messagebox.showwarning("AVISO", "Nenhuma ROI de tamanho livre foi selecionada. Selecione uma ROI primeiro.")

    def mostrar_histograma_da_img(self):
        if self.img is not None:
            plt.figure()
            plt.hist(self.img.ravel(), bins=self.bins, range=(0, 255), density=True, color='gray', edgecolor='black')
            plt.title("Histograma da imagem completa")
            plt.xlabel("Níveis de cinza")
            plt.ylabel("Densidade")
            plt.ylim(0, self.y_limit)
            plt.show()
        else:
            messagebox.showwarning("AVISO", "Nenhuma imagem foi carregada. Carregue uma imagem primeiro.")

    def ajustar_bins(self):
        new_bins = simpledialog.askinteger("Escala X (Bins)", "Digite o numero de bins:", minvalue=1, maxvalue=512)
        if new_bins:
            self.bins = new_bins

    def ajustar_y_limit(self):
        new_y_limit = simpledialog.askfloat("Escala Y (Limite)", "Digite o limite superior do eixo Y:", minvalue=0.001, maxvalue=1.0)
        if new_y_limit:
            self.y_limit = new_y_limit

    def mostrar_metricas_img(self):
        if self.img is not None:
            media = np.mean(self.img)
            mediana = np.median(self.img)
            desvio_pad = np.std(self.img)

            s = f"Média: {media:.2f}\nMediana: {mediana:.2f}\nDesvio padrão: {desvio_pad:.2f}"
            simpledialog.messagebox.showinfo("Métricas da imagem", s)
        else:
            messagebox.showwarning("AVISO", "Nenhuma imagem foi carregada. Carregue uma imagem primeiro.")

    def roi_28_x_28(self, pixel):
        # retorna a roi ajustada para o tamanho 28x28 e os pixels limites da roi

        x, y = pixel

        # garantir que a roi esteja dentro dos limites da imagem
        x_start = max(0, x - 14)
        y_start = max(0, y - 14)
        x_end = x_start + 28
        y_end = y_start + 28

        # ajustar se a roi ultrapassar as dimensoes da imagem
        if x_end > self.img.shape[1]:
            x_end = self.img.shape[1]
            x_start = x_end - 28
        if y_end > self.img.shape[0]:
            y_end = self.img.shape[0]
            y_start = y_end - 28

        roi = self.img[y_start:y_end, x_start:x_end]
        return roi, x_start, y_start, x_end, y_end

    def gerar_rois_manual(self):
        # geracao manual das rois para o figado e cortex renal tamanho fixo de 28x28
        if self.img is not None:
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

            # selecionar manuamente a posicao da roi do figado
            liver_pixel = self.get_pixel_do_clique(img_bgr, "Clique na posicao da ROI do Figado")
            if liver_pixel is None:
                messagebox.showwarning("AVISO", "Nenhum ponto foi selecionado para o figado.")
                return

            liver_roi, x_liver_start, y_liver_start, x_liver_end, y_liver_end = self.roi_28_x_28(liver_pixel)

            # selecionar manualmente a posicao da roi do rim
            kidney_pixel = self.get_pixel_do_clique(img_bgr, "Clique na posicao da ROI do Rim")
            if kidney_pixel is None:
                messagebox.showwarning("AVISO", "Nenhum ponto foi selecionado para o rim.")
                return

            kidney_roi, x_kidney_start, y_kidney_start, x_kidney_end, y_kidney_end = self.roi_28_x_28(kidney_pixel)

            # calcular o indice hepatorienal hi
            media_liver = np.mean(liver_roi)
            media_kidney = np.mean(kidney_roi)
            hi = media_liver / media_kidney if media_kidney != 0 else 1

            # Exibir o valor do HI
            messagebox.showinfo("Índice Hepatorrenal (HI)", f"O valor do HI é: {hi:.4f}")

            # normalizar a roi do figado
            liver_roi_ajustado = liver_roi * hi
            liver_roi_ajustado = np.round(liver_roi_ajustado).astype(np.uint8)

            # salvar a roi do figado
            nome_arquivo_roi = f"ROI_{self.current_patient:02d}_{self.current_image}.png"
            cv2.imwrite(nome_arquivo_roi, liver_roi_ajustado)
            messagebox.showinfo("SUCESSO", f"ROI do fígado salva como {nome_arquivo_roi}")

            # exibir os rois
            img_copy = img_bgr.copy()
            cv2.rectangle(img_copy, (x_liver_start, y_liver_start), (x_liver_end, y_liver_end), (0, 255, 0), 2)  # verde para o fígado
            cv2.rectangle(img_copy, (x_kidney_start, y_kidney_start), (x_kidney_end, y_kidney_end), (255, 0, 0), 2)  # azul para o rim

            img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("ROIs manuais: Figado (verde), Rim (azul)")
            plt.show()
        else:
            messagebox.showwarning("AVISO", "Nenhuma imagem foi carregada. Carregue uma imagem primeiro.")

    def gerar_rois_manual_dataset(self):
        if self.imagens is not None:
            # perguntar o pasciente e img inicial
            input_window = Toplevel(self.root)
            input_window.title("Especificar Ponto de Início")
            input_window.grab_set() #  focar nesta janela

            Label(input_window, text=f"Digite o número do paciente inicial (0-{self.total_patients - 1}):").pack(pady=5)
            paciente_selecionado = Entry(input_window)
            paciente_selecionado.pack(pady=5)

            Label(input_window, text=f"Digite o número da imagem inicial (0-{self.imagens_per_patient - 1}):").pack(pady=5)
            imagem_selecionada = Entry(input_window)
            imagem_selecionada.pack(pady=5)

            def iniciar_geracao():
                try:
                    num_paciente = int(paciente_selecionado.get())
                    num_imagem = int(imagem_selecionada.get())

                    # se foi o paciente e img inicial forem validos
                    if 0 <= num_paciente <= self.total_patients - 1 and 0 <= num_imagem <= self.imagens_per_patient - 1:
                        self.current_patient = num_paciente
                        self.current_image = num_imagem
                        input_window.destroy()

                        # abre o arquivo csv para escrita ou append
                        file_exists = os.path.isfile(self.csv_path)
                        if file_exists:
                            self.csv_file = open(
                                self.csv_path, "a", newline="", encoding="utf-8"
                            )
                            self.csv_writer = csv.DictWriter(
                                self.csv_file,
                                fieldnames=[
                                    "nome_arquivo",
                                    "classe",
                                    "figado_x",
                                    "figado_y",
                                    "rim_x",
                                    "rim_y",
                                    "HI",
                                    "coarseness",
                                    "contrast",
                                    "periodicity",
                                    "roughness",
                                    "entropia_d1",
                                    "entropia_d2",
                                    "entropia_d4",
                                    "entropia_d8",
                                    "homogeneidade_d1",
                                    "homogeneidade_d2",
                                    "homogeneidade_d4",
                                    "homogeneidade_d8",
                                ],
                                delimiter=";",
                            )
                        else:
                            self.csv_file = open(
                                self.csv_path, "w", newline="", encoding="utf-8"
                            )
                            self.csv_writer = csv.DictWriter(
                                self.csv_file,
                                fieldnames=[
                                    "nome_arquivo",
                                    "classe",
                                    "figado_x",
                                    "figado_y",
                                    "rim_x",
                                    "rim_y",
                                    "HI",
                                    "entropia",
                                    "homogeneidade",
                                    "coarseness",
                                    "contrast",
                                    "periodicity",
                                    "roughness",
                                ],
                                delimiter=";",
                            )
                            self.csv_writer.writeheader()

                        # seleciona primeira imagem
                        self.img = self.imagens[0][self.current_patient][self.current_image]
                        self.mostrar_img()
                        self.gerar_rois_manuais()
                    else:
                        messagebox.showerror("ERRO", "Números fora do intervalo válido.")
                except ValueError:
                    messagebox.showerror("ERRO", "Insira números inteiros válidos.")

            Button(input_window, text="Iniciar", command=iniciar_geracao).pack(pady=10)
        else:
            messagebox.showwarning("AVISO", "Nenhum banco de imagens carregado. Carregue um banco de imagens primeiro.")

    def gerar_rois_manuais(self):
        # funcao recursiva para fazer a geração manual das rois para o figado e rim para todas as imagens do datset
        # (a partir de um inicio dado)
        if self.img is not None:
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

            # selecionar manualmente a posicao da roi do figado
            liver_pixel = self.get_pixel_do_clique(img_bgr, f"Paciente {self.current_patient}, Imagem {self.current_image}: Clique na posicao da ROI do figado. Aperte ESC para sair")
            if liver_pixel is None:
                messagebox.showinfo("Processo Interrompido", "O processamento foi interrompido pelo usuário.")
                if self.csv_file:
                    self.csv_file.close()
                    self.csv_file = None
                return

            liver_roi, x_liver_start, y_liver_start, x_liver_end, y_liver_end = self.roi_28_x_28(liver_pixel)

            # selecionar manualmente a posisao da roi do rim
            kidney_pixel = self.get_pixel_do_clique(img_bgr, f"Paciente {self.current_patient}, Imagem {self.current_image}: Clique na posicao da ROI do rim. Aperte ESC para sair")
            if kidney_pixel is None:
                messagebox.showinfo("Processo Interrompido", "O processamento foi interrompido pelo usuário.")
                if self.csv_file:
                    self.csv_file.close()
                    self.csv_file = None
                return

            kidney_roi, x_kidney_start, y_kidney_start, x_kidney_end, y_kidney_end = self.roi_28_x_28(kidney_pixel)

            # calcular o indice hepatorenal hi
            media_liver = np.mean(liver_roi)
            media_kidney = np.mean(kidney_roi)
            hi = media_liver / media_kidney if media_kidney != 0 else 1

            # normalizar a roi do figado
            liver_roi_ajustado = liver_roi * hi
            liver_roi_ajustado = np.round(liver_roi_ajustado).astype(np.uint8)

            # salvar a roi do figado
            nome_arquivo_roi = f"ROI_{self.current_patient:02d}_{self.current_image}.png"
            cv2.imwrite(nome_arquivo_roi, liver_roi_ajustado)
            messagebox.showinfo("SUCESSO", f"ROI do fígado salva como {nome_arquivo_roi}")

            # exibir os rois
            img_copy = img_bgr.copy()
            cv2.rectangle(img_copy, (x_liver_start, y_liver_start), (x_liver_end, y_liver_end), (0, 255, 0), 2)  # verde para o fígado
            cv2.rectangle(img_copy, (x_kidney_start, y_kidney_start), (x_kidney_end, y_kidney_end), (255, 0, 0), 2)  # azul para o rim

            img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(f"ROIs Manuais: Fígado (verde), Rim (azul) - Paciente {self.current_patient}, Imagem {self.current_image}")
            plt.show()

            # calculo dos descritores de textura 
            entropias, homogeneidades, _ = self.calcular_descritores_glcm(liver_roi_ajustado, exibir_matrizes=False)
            
            # calculo sfn
            features, _ = pf.sfm_features(f=liver_roi_ajustado, mask=None, Lr=4, Lc=4)
            coarseness = features[0]
            contrast = features[1]
            periodicity = features[2]
            roughness = features[3]

            # determinar a clase do paciente
            class_label = 'Saudavel' if self.current_patient <= 16 else 'Esteatose'

            # remover virgulas e caracteres especiais dos dados
            nome_arquivo_roi_clean = re.sub(r'[^\w\-_\. ]', '_', nome_arquivo_roi)
            class_label_clean = re.sub(r'[^\w\-_\. ]', '_', class_label)

            # preparar os valores de entropia e homogeniedade para todas as distancias
            data_row = {
                'nome_arquivo': nome_arquivo_roi_clean,
                'classe': class_label_clean,
                'figado_x': x_liver_start,
                'figado_y': y_liver_start,
                'rim_x': x_kidney_start,
                'rim_y': y_kidney_start,
                'HI': f"{hi:.4f}",
                'coarseness': f"{coarseness:.4f}",
                'contrast': f"{contrast:.4f}",
                'periodicity': f"{periodicity:.4f}",
                'roughness': f"{roughness:.4f}",
                'entropia_d1': f"{entropias[0]:.4f}",
                'entropia_d2': f"{entropias[1]:.4f}",
                'entropia_d4': f"{entropias[2]:.4f}",
                'entropia_d8': f"{entropias[3]:.4f}",
                'homogeneidade_d1': f"{homogeneidades[0]:.4f}",
                'homogeneidade_d2': f"{homogeneidades[1]:.4f}",
                'homogeneidade_d4': f"{homogeneidades[2]:.4f}",
                'homogeneidade_d8': f"{homogeneidades[3]:.4f}"
            }

            # escrever linha no arquivo csv
            if self.csv_writer:
                self.csv_writer.writerow(data_row)
                self.csv_file.flush()  # garante que os dados sejam gravados imdiatamente

            # avançar para a prox imagem
            self.current_image += 1
            # CONDICOES DE PARADA

            # se acabou as imagens de um paciente
            if self.current_image >= self.imagens_per_patient:
                # passar pro prox
                self.current_patient += 1
                self.current_image = 0
                # verificar se chegou ao final do dataset
                if self.current_patient >= self.total_patients:
                    messagebox.showinfo("CONCLUIDO", "Processamento de todas as imagens concluído.")
                    if self.csv_file:
                        self.csv_file.close()
                        self.csv_file = None
                    return

            # carregar proxima img
            self.img = self.imagens[0][self.current_patient][self.current_image]
            self.mostrar_img()
            self.gerar_rois_manuais()
        else:
            messagebox.showwarning("AVISO", "Nenhuma imagem foi carregada. Carregue uma imagem primeiro.")

    def get_pixel_do_clique(self, img, window_title):
        pixel_do_clique = [] # usando um array pra ter a referencia na funcao mouse_callback

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pixel_do_clique.append((x, y))
                cv2.destroyWindow(window_title)

        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, mouse_callback)
        cv2.imshow(window_title, img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1: # tratamento caso fechou no X
                break
            if key == 27: # tecla esc
                self.parar_processo = True
                cv2.destroyWindow(window_title)
                break
            if len(pixel_do_clique) > 0:
                break

        if self.parar_processo:
            self.parar_processo = False  # resetar para futuras chamadas
            return None
        elif pixel_do_clique:
            return pixel_do_clique[0]
        else:
            return None
        
# ------------------------------------------------------------------------------------------------------------- Parte 2 - Classificação ----------------------------------------------------------
   # Parte 2 - Classificação
    def validacao_cruzada(self, X, y_encoded, pacientes, treinar_avaliar):
        print(f"Realizando validação cruzada Leave-One-Patient-Out com ordem aleatória de pacientes ({len(np.unique(pacientes))} pacientes).")
        seed = 42
        np.random.seed(seed)
        
        accuracies = []
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []
        matrizes_confusao = []
        histories = []
        model = None
        best_accuracy = 0.0
        fold_number = 1  # Para rastrear o número do fold

        # Cria um diretório para salvar as matrizes de confusão
        conf_matrix_dir = "matrizes_confusao"
        if not os.path.exists(conf_matrix_dir):
            os.makedirs(conf_matrix_dir)

        pacientes_ordem_aleatoria = list(np.unique(pacientes))
        random.shuffle(pacientes_ordem_aleatoria)

        for paciente_teste in pacientes_ordem_aleatoria:
            indice_test = np.where(pacientes == paciente_teste)[0]

            if len(indice_test) < 10:
                print(f"Paciente {paciente_teste} não possui imagens suficientes. Ignorando.")
                continue
            indice_test = np.random.choice(indice_test, size=10, replace=False)

            indices_train = np.where(pacientes != paciente_teste)[0]

            X_train, X_test = X[indices_train], X[indice_test]
            y_train, y_test = y_encoded[indices_train], y_encoded[indice_test]

            result = treinar_avaliar(X_train, X_test, y_train, y_test)

            accuracy = result['accuracy']
            sensitivity = result['sensitivity']
            specificity = result['specificity']
            precision = result['precision']
            f1_score = result['f1_score']
            matriz_confusao = result['matriz_confusao']
            if 'history' in result:
                history = result['history']
            else:
                history = None
            if 'model' in result:
                model = result['model']
            else:
                model = None

            accuracies.append(accuracy)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            precisions.append(precision)
            f1_scores.append(f1_score)
            matrizes_confusao.append(matriz_confusao)
            if 'history' in result:
                histories.append(history)

            # Salvar a matriz de confusão como imagem .png
            conf_matrix_filename = os.path.join(conf_matrix_dir, f"matriz_confusao_fold_{fold_number}.png")
            self.salvar_matriz_confusao_como_imagem(matriz_confusao, conf_matrix_filename)

            # Comparar e salvar o melhor modelo
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if model is not None:
                    # Salva o modelo apenas se ele existir
                    model.save('mobilenet_model.h5')

            print(f"Fold {fold_number} - Paciente {paciente_teste} (aleatório): Acurácia={accuracy:.4f}, Sensibilidade={sensitivity:.4f}, Especificidade={specificity:.4f}, Precisão={precision:.4f}, F1-score={f1_score:.4f}")

            fold_number += 1

        media_accuracy = np.mean(accuracies)
        media_sensitivity = np.mean(sensitivities)
        media_specificity = np.mean(specificities)
        media_precision = np.mean(precisions)
        media_f1_score = np.mean(f1_scores)
        return media_accuracy, media_sensitivity, media_specificity, media_precision, media_f1_score, matrizes_confusao, histories

    def salvar_matriz_confusao_como_imagem(self, matriz_confusao, filename):
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            matriz_confusao,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Esteatose', 'Saudável'],
            yticklabels=['Esteatose', 'Saudável']
        )
        plt.xlabel('Predição')
        plt.ylabel('Verdadeiro')
        plt.title(f'Matriz de Confusão')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Método para extrair números dos pacientes
    def extract_patient_numbers(self, data : pd.DataFrame):
        """
        Extrai os números dos pacientes a partir dos nomes dos arquivos.
        """
        def extract_patient_number(filename):
            match = re.match(r'ROI_(\d+)_\d+\.png', filename)
            if match:
                return int(match.group(1))
            else:
                return -1

        patient_numbers = data['nome_arquivo'].apply(extract_patient_number).values
        return patient_numbers

    def exibir_resultados(self, avg_accuracy, avg_sensitivity, avg_specificity, avg_precision, avg_f1_score, matrizes_confusao, label_encoder, modelo):
        result_window = Toplevel(self.root)
        result_window.title(f"Resultados da Classificação com {modelo}")

        # Mostrar dados médios
        s = f"Média de Acurácia: {avg_accuracy:.4f}\n"
        s += f"Média de Sensibilidade (Recall): {avg_sensitivity:.4f}\n"
        s += f"Média de Especificidade: {avg_specificity:.4f}\n"
        s += f"Média de Precisão: {avg_precision:.4f}\n"
        s += f"Média de F1-score: {avg_f1_score:.4f}\n\n"

        # Calcular métricas a partir da matriz de confusão acumulada
        total_matrizes_confusao = np.sum(matrizes_confusao, axis=0)
        tp, fp, fn, tn = total_matrizes_confusao.ravel()
        print(tn, fp, fn, tp)
        accuracy_accum = (tp + tn) / (tp + tn + fp + fn)
        sensitivity_accum = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity_accum = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision_accum = tp / (tp + fp) if (tp + fp) != 0 else 0
        f1_score_accum = (2 * precision_accum * sensitivity_accum) / (precision_accum + sensitivity_accum) if (precision_accum + sensitivity_accum) != 0 else 0

        # Mostrar métricas acumuladas
        s += "Métricas calculadas a partir da matriz de confusão acumulada:\n"
        s += f"Acurácia: {accuracy_accum:.4f}\n"
        s += f"Sensibilidade (Recall): {sensitivity_accum:.4f}\n"
        s += f"Especificidade: {specificity_accum:.4f}\n"
        s += f"Precisão: {precision_accum:.4f}\n"
        s += f"F1-score: {f1_score_accum:.4f}\n"
        Label(result_window, text=s).pack(pady=10)

        # Mostrar matriz de confusão
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            total_matrizes_confusao,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax
        )
        ax.set_xlabel('Predição')
        ax.set_ylabel('Verdadeiro')
        ax.set_title('Matriz de Confusão após Validação Cruzada')

        # Insere o gráfico na janela
        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        plt.close(fig)


    # Função para treinar e avaliar o SVM
    def classificar_com_svm(self, retornar_metricas=False):
        if not os.path.isfile('data.csv'):
            messagebox.showerror("Erro", "Arquivo 'data.csv' não encontrado. Por favor, gere o arquivo primeiro.")
            return None

        data = pd.read_csv('data.csv', delimiter=';')

        # Define as features e o target
        features = [
            'coarseness', 'contrast', 'periodicity', 'roughness',
            'entropia_d1', 'entropia_d2', 'entropia_d4', 'entropia_d8',
            'homogeneidade_d1', 'homogeneidade_d2', 'homogeneidade_d4', 'homogeneidade_d8'
        ]
        X = data[features].astype(float).values
        y = data['classe'].values

        # Codifica as labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Extrai os números dos pacientes dos nomes dos arquivos
        patient_numbers = self.extract_patient_numbers(data)

        # Medir o tempo de execução
        start_time = time.time()

        # Executa a validação cruzada
        avg_accuracy, avg_sensitivity, avg_specificity, avg_precision, avg_f1_score, conf_matrices, _ = self.validacao_cruzada(
            X, y_encoded, patient_numbers, self.treinar_avaliar_svm
        )

        execution_time = time.time() - start_time  # Tempo de execução

        if retornar_metricas:
            return avg_accuracy, avg_sensitivity, avg_specificity, avg_precision, avg_f1_score, conf_matrices, execution_time

        # Exibe os resultados
        self.exibir_resultados(
            avg_accuracy, avg_sensitivity, avg_specificity, avg_precision, avg_f1_score, conf_matrices, le, modelo="SVM"
        )

    def treinar_avaliar_mobilenet(self, X_train, X_test, y_train, y_test):
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

        # Pre-processamento das imagens
        X_train = preprocess_input(X_train)
        X_test = preprocess_input(X_test)

        # Criação do modelo com MobileNet pré-treinado
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Descongela as camadas para fine-tuning, se especificado
        if self.parametros_mobilenet['fine_tune_layers'] > 0:
            for layer in base_model.layers[-self.parametros_mobilenet['fine_tune_layers']:]:
                layer.trainable = True
        else:
            for layer in base_model.layers:
                layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.parametros_mobilenet['dropout_rate'])(x)
        x = Dense(1024, activation=self.parametros_mobilenet['activation_function'])(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Configura o otimizador com a taxa de aprendizado especificada
        optimizer_name = self.parametros_mobilenet['optimizer']
        learning_rate = self.parametros_mobilenet['learning_rate']

        if optimizer_name.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.parametros_mobilenet.get('momentum', 0.0)
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        else:
            optimizer = tf.keras.optimizers.get(optimizer_name)
            optimizer.learning_rate = learning_rate  # Define a taxa de aprendizado

        # Compila o modelo usando os parâmetros definidos
        model.compile(
            optimizer=optimizer,
            loss=self.parametros_mobilenet['loss_function'],
            metrics=['accuracy']
        )

        # Define callbacks com o patience especificado
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.parametros_mobilenet['early_stopping_patience'],
                restore_best_weights=True
            )
        ]

        # Treinamento do modelo usando os parâmetros definidos
        history = model.fit(
            X_train, y_train,
            epochs=self.parametros_mobilenet['epochs'],
            batch_size=self.parametros_mobilenet['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        # Faz predições no conjunto de teste
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

        # Calcula a matriz de confusão
        matriz_confusao = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matriz_confusao.ravel()

        # Calcula as métricas a partir da matriz de confusão
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Sensibilidade ou Recall
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

        result = dict()
        result['accuracy'] = accuracy
        result['sensitivity'] = sensitivity
        result['specificity'] = specificity
        result['precision'] = precision
        result['f1_score'] = f1
        result['matriz_confusao'] = matriz_confusao
        result['history'] = history
        result['model'] = model  # Retorna o modelo

        return result

    def classificar_com_mobilenet(self, retornar_metricas=False):
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        if not os.path.isfile('data.csv'):
            messagebox.showerror("Erro", "Arquivo 'data.csv' não encontrado. Por favor, gere o arquivo primeiro.")
            return None

        data = pd.read_csv('data.csv', delimiter=';')
        image_files = data['nome_arquivo'].values
        y = data['classe'].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        patient_numbers = self.extract_patient_numbers(data)

        image_dir = "ROIS"
        images = []
        for img_file in image_files:
            full_path = os.path.join(image_dir, img_file)
            if os.path.isfile(full_path):
                img = load_img(full_path, target_size=(224, 224))
                img_array = img_to_array(img)
                images.append(img_array)
            else:
                messagebox.showerror("Erro", f"Arquivo {full_path} não encontrado.")
                return None

        X = np.array(images)
        y_encoded = np.array(y_encoded)

        start_time = time.time()

        avg_accuracy, avg_sensitivity, avg_specificity, avg_precision, avg_f1_score, conf_matrices, histories = self.validacao_cruzada(
            X, y_encoded, patient_numbers, self.treinar_avaliar_mobilenet
        )

        execution_time = time.time() - start_time  # Tempo de execução

        if retornar_metricas:
            return avg_accuracy, avg_sensitivity, avg_specificity, avg_precision, avg_f1_score, conf_matrices, execution_time, histories

        self.exibir_resultados(
            avg_accuracy, avg_sensitivity, avg_specificity, avg_precision, avg_f1_score, conf_matrices, le, modelo="MobileNet"
        )

        self.plot_learning_curves(histories)

    def plot_learning_curves(self, histories):
        # Inicializa as listas para armazenar as métricas por época
        max_epochs = max([len(history.history['accuracy']) for history in histories])
        num_folds = len(histories)
        train_acc_epochs = np.full((max_epochs, num_folds), np.nan)
        val_acc_epochs = np.full((max_epochs, num_folds), np.nan)

        for fold_idx, history in enumerate(histories):
            num_epochs_fold = len(history.history['accuracy'])
            train_acc_epochs[:num_epochs_fold, fold_idx] = history.history['accuracy']
            val_acc_epochs[:num_epochs_fold, fold_idx] = history.history['val_accuracy']

        # Calcula o número de épocas treinadas em cada fold
        num_epochs_per_fold = np.sum(~np.isnan(train_acc_epochs), axis=0)

        # Determina o número máximo de épocas treinadas em qualquer fold
        num_epochs_eff = int(np.max(num_epochs_per_fold))
        epochs_range = range(1, num_epochs_eff + 1)

        # Calcula as métricas médias por época, ignorando NaNs
        avg_train_acc = np.nanmean(train_acc_epochs[:num_epochs_eff, :], axis=1)
        avg_val_acc = np.nanmean(val_acc_epochs[:num_epochs_eff, :], axis=1)

        # Cria uma nova janela Tkinter para exibir o gráfico
        graph_window = Toplevel(self.root)
        graph_window.title("Gráfico de Aprendizado - MobileNet")

        # Plota as curvas de aprendizado
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs_range, avg_train_acc, label='Acurácia de Treino')
        ax.plot(epochs_range, avg_val_acc, label='Acurácia de Validação')
        ax.legend(loc='lower right')
        ax.set_title('Acurácia Média de Treino e Validação por Época')
        ax.set_xlabel('Épocas')
        ax.set_ylabel('Acurácia')
        plt.tight_layout()

        # Insere o gráfico na janela Tkinter
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        plt.close(fig)  # Fecha a figura para liberar memória

    def classificar_e_comparar(self):
        # Executa a classificação com SVM
        svm_results = self.classificar_com_svm(retornar_metricas=True)

        if not svm_results:
            messagebox.showerror("Erro", "Erro na classificação com SVM.")
            return

        avg_accuracy_svm, avg_sensitivity_svm, avg_specificity_svm, avg_precision_svm, avg_f1_score_svm, conf_matrix_svm, execution_time_svm = svm_results

        # Executa a classificação com MobileNet
        mobilenet_results = self.classificar_com_mobilenet(retornar_metricas=True)

        if not mobilenet_results:
            messagebox.showerror("Erro", "Erro na classificação com MobileNet.")
            return

        avg_accuracy_mobilenet, avg_sensitivity_mobilenet, avg_specificity_mobilenet, avg_precision_mobilenet, avg_f1_score_mobilenet, conf_matrix_mobilenet, execution_time_mobilenet, histories_mobilenet = mobilenet_results

        # Exibe a tabela comparativa
        self.exibir_tabela_comparativa(
            avg_accuracy_svm, avg_sensitivity_svm, avg_specificity_svm, avg_precision_svm, avg_f1_score_svm, conf_matrix_svm,
            avg_accuracy_mobilenet, avg_sensitivity_mobilenet, avg_specificity_mobilenet, avg_precision_mobilenet, avg_f1_score_mobilenet, conf_matrix_mobilenet,
            histories_mobilenet, execution_time_svm=execution_time_svm, execution_time_mobilenet=execution_time_mobilenet
        )


    def treinar_avaliar_svm(self, X_train, X_test, y_train, y_test):
        # Parâmetros do SVM
        svm_params = {
            'kernel': self.parametros_svm['kernel'],
            'C': self.parametros_svm['C'],
            'gamma': self.parametros_svm['gamma'],
            'degree': self.parametros_svm['degree'],
            'coef0': self.parametros_svm['coef0'],
            'class_weight': self.parametros_svm['class_weight'],
            'decision_function_shape': self.parametros_svm['decision_function_shape']
        }

        # Ajustar o parâmetro 'class_weight' se necessário
        if svm_params['class_weight'] == 'None' or svm_params['class_weight'] == '':
            svm_params['class_weight'] = None
        elif svm_params['class_weight'] == 'balanced':
            svm_params['class_weight'] = 'balanced'

        classificador_svm = SVC(**svm_params)
        classificador_svm.fit(X_train, y_train)

        y_pred = classificador_svm.predict(X_test)

        # Calcula a matriz de confusão
        matriz_confusao = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = matriz_confusao.ravel()

        # Calcula as métricas a partir da matriz de confusão
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Sensibilidade ou Recall
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

        result = dict()
        result['accuracy'] = accuracy
        result['sensitivity'] = sensitivity
        result['specificity'] = specificity
        result['precision'] = precision
        result['f1_score'] = f1
        result['matriz_confusao'] = matriz_confusao

        return result

    def exibir_tabela_comparativa(
    self, avg_accuracy_svm, avg_sensitivity_svm, avg_specificity_svm, avg_precision_svm, avg_f1_score_svm, conf_matrices_svm,
    avg_accuracy_mobilenet, avg_sensitivity_mobilenet, avg_specificity_mobilenet, avg_precision_mobilenet, avg_f1_score_mobilenet, conf_matrices_mobilenet,
    histories_mobilenet, execution_time_svm=None, execution_time_mobilenet=None
):
        # Cria uma janela para exibir os resultados
        result_window = Toplevel(self.root)
        result_window.title("Comparação de Classificadores")

        # Cria uma tabela usando o módulo ttk
        from tkinter import ttk

        # Agrega as matrizes de confusão
        conf_matrices_svm = np.array(conf_matrices_svm)
        conf_matrix_svm = np.sum(conf_matrices_svm, axis=0)

        conf_matrices_mobilenet = np.array(conf_matrices_mobilenet)
        conf_matrix_mobilenet = np.sum(conf_matrices_mobilenet, axis=0)

        # Calcular métricas a partir das matrizes de confusão acumuladas
        tn_svm, fp_svm, fn_svm, tp_svm = conf_matrix_svm.ravel()
        accuracy_svm_accum = (tp_svm + tn_svm) / (tp_svm + tn_svm + fp_svm + fn_svm)
        sensitivity_svm_accum = tp_svm / (tp_svm + fn_svm) if (tp_svm + fn_svm) != 0 else 0
        specificity_svm_accum = tn_svm / (tn_svm + fp_svm) if (tn_svm + fp_svm) != 0 else 0
        precision_svm_accum = tp_svm / (tp_svm + fp_svm) if (tp_svm + fp_svm) != 0 else 0
        f1_score_svm_accum = (2 * precision_svm_accum * sensitivity_svm_accum) / (precision_svm_accum + sensitivity_svm_accum) if (precision_svm_accum + sensitivity_svm_accum) != 0 else 0

        tn_mob, fp_mob, fn_mob, tp_mob = conf_matrix_mobilenet.ravel()
        accuracy_mob_accum = (tp_mob + tn_mob) / (tp_mob + tn_mob + fp_mob + fn_mob)
        sensitivity_mob_accum = tp_mob / (tp_mob + fn_mob) if (tp_mob + fn_mob) != 0 else 0
        specificity_mob_accum = tn_mob / (tn_mob + fp_mob) if (tn_mob + fp_mob) != 0 else 0
        precision_mob_accum = tp_mob / (tp_mob + fp_mob) if (tp_mob + fp_mob) != 0 else 0
        f1_score_mob_accum = (2 * precision_mob_accum * sensitivity_mob_accum) / (precision_mob_accum + sensitivity_mob_accum) if (precision_mob_accum + sensitivity_mob_accum) != 0 else 0

        # Dados para a tabela
        metrics = [
            "Acurácia Média",
            "Acurácia (Matriz Acumulada)",
            "Sensibilidade Média",
            "Sensibilidade (Matriz Acumulada)",
            "Especificidade Média",
            "Especificidade (Matriz Acumulada)",
            "Precisão Média",
            "Precisão (Matriz Acumulada)",
            "F1-score Médio",
            "F1-score (Matriz Acumulada)",
            "Tempo de Execução (segundos)"
        ]
        svm_values = [
            f"{avg_accuracy_svm:.4f}",
            f"{accuracy_svm_accum:.4f}",
            f"{avg_sensitivity_svm:.4f}",
            f"{sensitivity_svm_accum:.4f}",
            f"{avg_specificity_svm:.4f}",
            f"{specificity_svm_accum:.4f}",
            f"{avg_precision_svm:.4f}",
            f"{precision_svm_accum:.4f}",
            f"{avg_f1_score_svm:.4f}",
            f"{f1_score_svm_accum:.4f}",
            f"{execution_time_svm:.2f}" if execution_time_svm is not None else "N/A"
        ]
        mobilenet_values = [
            f"{avg_accuracy_mobilenet:.4f}",
            f"{accuracy_mob_accum:.4f}",
            f"{avg_sensitivity_mobilenet:.4f}",
            f"{sensitivity_mob_accum:.4f}",
            f"{avg_specificity_mobilenet:.4f}",
            f"{specificity_mob_accum:.4f}",
            f"{avg_precision_mobilenet:.4f}",
            f"{precision_mob_accum:.4f}",
            f"{avg_f1_score_mobilenet:.4f}",
            f"{f1_score_mob_accum:.4f}",
            f"{execution_time_mobilenet:.2f}" if execution_time_mobilenet is not None else "N/A"
        ]

        # Configura a tabela
        tree = ttk.Treeview(result_window, columns=("Métrica", "SVM", "MobileNet"), show='headings')
        tree.heading("Métrica", text="Métrica")
        tree.heading("SVM", text="SVM")
        tree.heading("MobileNet", text="MobileNet")

        # Insere os dados na tabela
        for metric, svm_val, mobilenet_val in zip(metrics, svm_values, mobilenet_values):
            tree.insert("", "end", values=(metric, svm_val, mobilenet_val))

        tree.pack(pady=10)

        # Cria um notebook para organizar os gráficos
        notebook = ttk.Notebook(result_window)
        notebook.pack(expand=True, fill='both')

        # Frame para as matrizes de confusão
        conf_matrix_frame = Frame(notebook)
        notebook.add(conf_matrix_frame, text='Matrizes de Confusão')

        # Frame para o gráfico de aprendizado
        learning_curve_frame = Frame(notebook)
        notebook.add(learning_curve_frame, text='Gráfico de Aprendizado')

        # Exibe as matrizes de confusão lado a lado
        fig_cm, axes_cm = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0])
        axes_cm[0].set_title('Matriz de Confusão - SVM')
        axes_cm[0].set_xlabel('Predição')
        axes_cm[0].set_ylabel('Verdadeiro')

        sns.heatmap(conf_matrix_mobilenet, annot=True, fmt='d', cmap='Blues', ax=axes_cm[1])
        axes_cm[1].set_title('Matriz de Confusão - MobileNet')
        axes_cm[1].set_xlabel('Predição')
        axes_cm[1].set_ylabel('Verdadeiro')

        # Insere o gráfico das matrizes de confusão na aba correspondente
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=conf_matrix_frame)
        canvas_cm.draw()
        canvas_cm.get_tk_widget().pack()

        # Fecha a figura para liberar memória
        plt.close(fig_cm)

        # Plota o gráfico de aprendizado do MobileNet usando a função existente
        fig_lc = self.plot_learning_curves(histories_mobilenet)

        # Insere o gráfico de aprendizado na aba correspondente
        canvas_lc = FigureCanvasTkAgg(fig_lc, master=learning_curve_frame)
        canvas_lc.draw()
        canvas_lc.get_tk_widget().pack()

        # Fecha a figura para liberar memória
        plt.close(fig_lc)

    def executar_modelo_salvo(self):
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        from tensorflow.keras.applications.mobilenet import preprocess_input

        if not os.path.exists('mobilenet_model.h5'):
            messagebox.showerror("Erro", "O modelo salvo não foi encontrado. Por favor, treine e salve o modelo primeiro.")
            return

        # Solicita ao usuário que selecione uma imagem
        image_path = filedialog.askopenfilename(title="Selecione uma imagem para classificação", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not image_path:
            return  # Usuário cancelou

        # Carrega e processa a imagem selecionada
        try:
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão de lote
            img_array = preprocess_input(img_array)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar a imagem: {e}")
            return

        # Carrega o modelo salvo
        try:
            model = tf.keras.models.load_model('mobilenet_model.h5')
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar o modelo salvo: {e}")
            return

        # Realiza a predição
        try:
            y_pred_prob = model.predict(img_array)
            y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao fazer a previsão: {e}")
            return

        # Mapeia a classe predita para o rótulo correspondente
        class_mapping = {1: 'Saudavel', 0: 'Esteatose'}
        class_label = class_mapping.get(y_pred[0], 'Desconhecido')

        # Exibe o resultado
        messagebox.showinfo("Resultado da Classificação", f"A imagem foi classificada como: {class_label}")

    # Método para o Menu de Parâmetros
    def menu_de_parametros(self):
        # Cria uma janela para os parâmetros
        param_window = Toplevel(self.root)
        param_window.title("Configurar Parâmetros")
        param_window.grab_set()

        # Cria uma barra de menu para a janela de parâmetros
        param_menu_bar = Menu(param_window)
        param_window.config(menu=param_menu_bar)

        # Cria o menu "Ajuda"
        help_menu = Menu(param_menu_bar, tearoff=0)
        param_menu_bar.add_cascade(label="Ajuda", menu=help_menu)

        # Dicionário com as informações dos parâmetros
        parameters_info = {
            'SVM': {
                'Kernel': 'Define a função kernel usada pelo SVM.\nOpções:\n- linear: Kernel linear.\n- poly: Kernel polinomial.\n- rbf: Função de base radial.\n- sigmoid: Função sigmoide.',
                'C': 'Parâmetro de regularização que controla o trade-off entre maximizar a margem e minimizar o erro de classificação.\nValores maiores enfatizam a minimização do erro no treinamento.',
                'Gamma': 'Coeficiente do kernel para kernels "rbf", "poly" e "sigmoid".\nOpções:\n- scale: 1 / (n_features * X.var())\n- auto: 1 / n_features\n- Também pode ser um valor numérico.',
                'Degree': 'Grau do kernel polinomial ("poly"). Especifica o grau do polinômio.',
                'Coef0': 'Termo independente em kernels polinomiais e sigmoides.',
                'Class Weight': 'Define os pesos das classes para lidar com desbalanceamento.\nOpções:\n- None: Nenhum peso especial.\n- balanced: Ajusta os pesos inversamente proporcionais às frequências das classes.',
                'Decision Function Shape': 'Específico para problemas multiclasse.\nOpções:\n- ovr: One-vs-Rest.\n- ovo: One-vs-One.'
            },
            'MobileNet': {
                'Número de Épocas': 'Número de vezes que o algoritmo irá percorrer todo o conjunto de treinamento.',
                'Batch Size': 'Número de amostras que serão propagadas através da rede antes de atualizar os pesos.',
                'Otimizador': 'Algoritmo usado para atualizar os pesos da rede neural.\nOpções comuns:\n- adam\n- sgd\n- rmsprop',
                'Learning Rate': 'Taxa de aprendizado do otimizador. Controla o tamanho dos passos na otimização.',
                'Fine-tune Layers': 'Número de camadas finais do modelo base que serão descongeladas para treinamento (fine-tuning).',
                'Early Stopping Patience': 'Número de épocas sem melhoria na perda de validação antes de parar o treinamento.',
                'Loss Function': 'Função de perda utilizada para calcular o erro.\nOpções:\n- binary_crossentropy\n- categorical_crossentropy\n- etc.',
                'Dropout Rate': 'Taxa de dropout aplicada para prevenir overfitting. Valor entre 0 e 1.',
                'Activation Function': 'Função de ativação usada nas camadas densas.\nOpções comuns:\n- relu\n- sigmoid\n- tanh',
                'Momentum': 'Parâmetro do otimizador SGD que acelera o gradiente em direção ao mínimo global.'
            }
        }

        # Função para exibir a explicação do parâmetro selecionado
        def show_parameter_info(model, param):
            info = parameters_info[model][param]
            messagebox.showinfo(f"Ajuda - {model} - {param}", info)

        # Adiciona submenus para SVM e MobileNet no menu de Ajuda
        svm_help_menu = Menu(help_menu, tearoff=0)
        mobilenet_help_menu = Menu(help_menu, tearoff=0)
        help_menu.add_cascade(label="SVM", menu=svm_help_menu)
        help_menu.add_cascade(label="MobileNet", menu=mobilenet_help_menu)

        # Adiciona os parâmetros do SVM ao submenu de Ajuda
        for param in parameters_info['SVM']:
            svm_help_menu.add_command(
                label=param,
                command=lambda p=param: show_parameter_info('SVM', p)
            )

        # Adiciona os parâmetros do MobileNet ao submenu de Ajuda
        for param in parameters_info['MobileNet']:
            mobilenet_help_menu.add_command(
                label=param,
                command=lambda p=param: show_parameter_info('MobileNet', p)
            )

        # Cria abas para SVM e MobileNet
        from tkinter import ttk
        notebook = ttk.Notebook(param_window)
        notebook.pack(expand=True, fill='both')

        # Parâmetros do SVM
        svm_frame = Frame(notebook)
        notebook.add(svm_frame, text='SVM')

        # Parâmetros do MobileNet
        mobilenet_frame = Frame(notebook)
        notebook.add(mobilenet_frame, text='MobileNet')

        # --- Parâmetros do SVM ---
        Label(svm_frame, text="Kernel:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        svm_kernel = StringVar(value=self.parametros_svm.get('kernel', 'linear'))
        Entry(svm_frame, textvariable=svm_kernel).grid(row=0, column=1, padx=5, pady=5)

        Label(svm_frame, text="C:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        svm_c = DoubleVar(value=self.parametros_svm.get('C', 1.0))
        Entry(svm_frame, textvariable=svm_c).grid(row=1, column=1, padx=5, pady=5)

        Label(svm_frame, text="Gamma:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        svm_gamma = StringVar(value=self.parametros_svm.get('gamma', 'scale'))
        Entry(svm_frame, textvariable=svm_gamma).grid(row=2, column=1, padx=5, pady=5)

        Label(svm_frame, text="Degree (para kernel 'poly'):").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        svm_degree = IntVar(value=self.parametros_svm.get('degree', 3))
        Entry(svm_frame, textvariable=svm_degree).grid(row=3, column=1, padx=5, pady=5)

        Label(svm_frame, text="Coef0 (para kernels 'poly' e 'sigmoid'):").grid(row=4, column=0, padx=5, pady=5, sticky='e')
        svm_coef0 = DoubleVar(value=self.parametros_svm.get('coef0', 0.0))
        Entry(svm_frame, textvariable=svm_coef0).grid(row=4, column=1, padx=5, pady=5)

        Label(svm_frame, text="Class Weight:").grid(row=5, column=0, padx=5, pady=5, sticky='e')
        svm_class_weight = StringVar(value=self.parametros_svm.get('class_weight', 'None'))
        Entry(svm_frame, textvariable=svm_class_weight).grid(row=5, column=1, padx=5, pady=5)

        Label(svm_frame, text="Decision Function Shape:").grid(row=6, column=0, padx=5, pady=5, sticky='e')
        svm_decision_function_shape = StringVar(value=self.parametros_svm.get('decision_function_shape', 'ovr'))
        Entry(svm_frame, textvariable=svm_decision_function_shape).grid(row=6, column=1, padx=5, pady=5)

        # --- Parâmetros do MobileNet ---
        Label(mobilenet_frame, text="Número de Épocas:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        mobilenet_epochs = IntVar(value=self.parametros_mobilenet.get('epochs', 5))
        Entry(mobilenet_frame, textvariable=mobilenet_epochs).grid(row=0, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Batch Size:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        mobilenet_batch_size = IntVar(value=self.parametros_mobilenet.get('batch_size', 16))
        Entry(mobilenet_frame, textvariable=mobilenet_batch_size).grid(row=1, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Otimizador:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        mobilenet_optimizer = StringVar(value=self.parametros_mobilenet.get('optimizer', 'adam'))
        Entry(mobilenet_frame, textvariable=mobilenet_optimizer).grid(row=2, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Learning Rate:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        mobilenet_learning_rate = DoubleVar(value=self.parametros_mobilenet.get('learning_rate', 0.001))
        Entry(mobilenet_frame, textvariable=mobilenet_learning_rate).grid(row=3, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Fine-tune Layers (Descongelar N camadas):").grid(row=4, column=0, padx=5, pady=5, sticky='e')
        mobilenet_fine_tune_layers = IntVar(value=self.parametros_mobilenet.get('fine_tune_layers', 0))
        Entry(mobilenet_frame, textvariable=mobilenet_fine_tune_layers).grid(row=4, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Early Stopping Patience:").grid(row=5, column=0, padx=5, pady=5, sticky='e')
        mobilenet_early_stopping_patience = IntVar(value=self.parametros_mobilenet.get('early_stopping_patience', 3))
        Entry(mobilenet_frame, textvariable=mobilenet_early_stopping_patience).grid(row=5, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Loss Function:").grid(row=6, column=0, padx=5, pady=5, sticky='e')
        mobilenet_loss_function = StringVar(value=self.parametros_mobilenet.get('loss_function', 'binary_crossentropy'))
        Entry(mobilenet_frame, textvariable=mobilenet_loss_function).grid(row=6, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Dropout Rate:").grid(row=7, column=0, padx=5, pady=5, sticky='e')
        mobilenet_dropout_rate = DoubleVar(value=self.parametros_mobilenet.get('dropout_rate', 0.0))
        Entry(mobilenet_frame, textvariable=mobilenet_dropout_rate).grid(row=7, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Activation Function:").grid(row=8, column=0, padx=5, pady=5, sticky='e')
        mobilenet_activation_function = StringVar(value=self.parametros_mobilenet.get('activation_function', 'relu'))
        Entry(mobilenet_frame, textvariable=mobilenet_activation_function).grid(row=8, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Momentum (para SGD):").grid(row=9, column=0, padx=5, pady=5, sticky='e')
        mobilenet_momentum = DoubleVar(value=self.parametros_mobilenet.get('momentum', 0.0))
        Entry(mobilenet_frame, textvariable=mobilenet_momentum).grid(row=9, column=1, padx=5, pady=5)

        # Botão para salvar as configurações
        def salvar_parametros():
            # Atualiza os parâmetros do SVM
            self.parametros_svm['kernel'] = svm_kernel.get()
            self.parametros_svm['C'] = svm_c.get()
            self.parametros_svm['gamma'] = svm_gamma.get()
            self.parametros_svm['degree'] = svm_degree.get()
            self.parametros_svm['coef0'] = svm_coef0.get()
            class_weight_value = svm_class_weight.get()
            self.parametros_svm['class_weight'] = None if class_weight_value == 'None' else class_weight_value
            self.parametros_svm['decision_function_shape'] = svm_decision_function_shape.get()

            # Atualiza os parâmetros do MobileNet
            self.parametros_mobilenet['epochs'] = mobilenet_epochs.get()
            self.parametros_mobilenet['batch_size'] = mobilenet_batch_size.get()
            self.parametros_mobilenet['optimizer'] = mobilenet_optimizer.get()
            self.parametros_mobilenet['learning_rate'] = mobilenet_learning_rate.get()
            self.parametros_mobilenet['fine_tune_layers'] = mobilenet_fine_tune_layers.get()
            self.parametros_mobilenet['early_stopping_patience'] = mobilenet_early_stopping_patience.get()
            self.parametros_mobilenet['loss_function'] = mobilenet_loss_function.get()
            self.parametros_mobilenet['dropout_rate'] = mobilenet_dropout_rate.get()
            self.parametros_mobilenet['activation_function'] = mobilenet_activation_function.get()
            self.parametros_mobilenet['momentum'] = mobilenet_momentum.get()

            messagebox.showinfo("Parâmetros Salvos", "Os parâmetros foram atualizados com sucesso.")
            param_window.destroy()

        Button(param_window, text="Salvar", command=salvar_parametros).pack(pady=10)

# iniciar o aplicativo
root = Tk()
app = App(root)
app.mainloop()