# Trabalho Prático - Diagnóstico de Esteatose Hepática em Exames de Ultrassom
# Integrantes(Matricula): Felipe Campolina(762732), Leandro Guido(777801) e Marcelo Augusto(775119)
import os
import csv
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, Menu, filedialog, simpledialog, messagebox, Toplevel, StringVar, Radiobutton, Entry, Frame
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
import re
from tkinter import ttk
from tkinter import DoubleVar, IntVar

# variaveis para facilitar debug
CARREGAR_DATASET_AUTOMATICO = False

class App(Frame):
    def __init__(self, root:Tk):
        super().__init__(root)

        self.root = root
        self.root.title("Visualizador de imagens em tons de cinza com ROIs")
        self.root.geometry("800x600")

        # criando barra de menu
        self.menu_bar = Menu(root)
        root.config(menu=self.menu_bar)
           # Parâmetros padrão para o SVM
        # Parâmetros padrão para o SVM
        self.svm_params = {
            'kernel': 'linear',
            'C': 1.0,
            'gamma': 'scale',
            'degree': 3,
            'coef0': 0.0,
            'class_weight': None,
            'decision_function_shape': 'ovr'
        }

        # Parâmetros padrão para o MobileNet
        self.mobilenet_params = {
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
        self.csv_file = None
        self.csv_writer = None
        self.csv_file_path = "data.csv"  # nome fixo para o arquivo CSV

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
                        file_exists = os.path.isfile(self.csv_file_path)
                        if file_exists:
                            self.csv_file = open(
                                self.csv_file_path, "a", newline="", encoding="utf-8"
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
                                self.csv_file_path, "w", newline="", encoding="utf-8"
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

    # Função genérica de validação cruzada
    def cross_validate_model(self, X, y_encoded, patient_numbers, train_and_evaluate_func, *args, **kwargs):
        """
        Realiza a validação cruzada leave-one-patient-out.

        :param X: Features ou imagens de entrada.
        :param y_encoded: Labels codificados.
        :param patient_numbers: Números dos pacientes correspondentes às amostras.
        :param train_and_evaluate_func: Função que treina e avalia o modelo.
        :param args: Argumentos adicionais para a função de treinamento e avaliação.
        :param kwargs: Argumentos nomeados adicionais para a função de treinamento e avaliação.
        :return: Métricas médias, matrizes de confusão e, opcionalmente, históricos de treinamento.
        """
        unique_patients = np.unique(patient_numbers)
        print(f"Realizando validação cruzada leave-one-patient-out com {len(unique_patients)} pacientes.")

        # Inicializa as listas para armazenar as métricas
        accuracies = []
        sensitivities = []
        specificities = []
        conf_matrices = []
        histories = []

        for test_patient in unique_patients:
            # Índices do paciente atual
            test_indices = np.where(patient_numbers == test_patient)[0]

            # Verifica se o paciente tem pelo menos 10 imagens
            if len(test_indices) < 10:
                print(f"Paciente {test_patient} não possui imagens suficientes. Ignorando.")
                continue

            # Seleciona as primeiras 10 imagens para o teste
            test_indices = test_indices[:10]

            # Índices de treinamento são todos os outros
            train_indices = np.where(patient_numbers != test_patient)[0]

            # Divide os dados
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

            # Treina e avalia o modelo usando a função fornecida
            result = train_and_evaluate_func(X_train, X_test, y_train, y_test, *args, **kwargs)

            # Desempacota os resultados
            acc, sensitivity, specificity, cm = result[:4]
            history = result[4] if len(result) > 4 else None

            # Armazena as métricas
            accuracies.append(acc)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            conf_matrices.append(cm)
            if history is not None:
                histories.append(history)

            print(f"Paciente {test_patient}: Acurácia={acc:.4f}, Sensibilidade={sensitivity:.4f}, Especificidade={specificity:.4f}")

        # Calcula as métricas médias
        avg_accuracy, avg_sensitivity, avg_specificity = self.calculate_average_metrics(accuracies, sensitivities, specificities)

        return avg_accuracy, avg_sensitivity, avg_specificity, conf_matrices, histories

    # Método para extrair números dos pacientes
    def extract_patient_numbers(self, data):
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

    # Método para calcular as métricas médias
    def calculate_average_metrics(self, accuracies, sensitivities, specificities):
        """
        Calcula as métricas médias.
        """
        avg_accuracy = np.mean(accuracies)
        avg_sensitivity = np.mean(sensitivities)
        avg_specificity = np.mean(specificities)
        return avg_accuracy, avg_sensitivity, avg_specificity

    # Método para exibir os resultados
    def display_classification_results(self, avg_accuracy, avg_sensitivity, avg_specificity, conf_matrices, label_encoder, model_name):
        """
        Exibe os resultados da classificação em uma janela.
        """
        # Soma as matrizes de confusão
        total_conf_matrix = np.sum(conf_matrices, axis=0)

        # Exibe os resultados em uma janela
        result_window = Toplevel(self.root)
        result_window.title(f"Resultados da Classificação com {model_name}")

        result_text = f"Média de Acurácia: {avg_accuracy:.4f}\n"
        result_text += f"Média de Sensibilidade: {avg_sensitivity:.4f}\n"
        result_text += f"Média de Especificidade: {avg_specificity:.4f}\n"
        Label(result_window, text=result_text).pack(pady=10)

        # Exibe a matriz de confusão
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            total_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax
        )
        ax.set_xlabel('Predição')
        ax.set_ylabel('Verdadeiro')
        ax.set_title('Matriz de Confusão após Validação Cruzada')

        # Insere o gráfico na janela Tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Fecha a figura para liberar memória
        plt.close(fig)

    # Função para treinar e avaliar o SVM
    def train_and_evaluate_svm(self, X_train, X_test, y_train, y_test):
        from sklearn.svm import SVC
        from sklearn.metrics import confusion_matrix, accuracy_score

        # Prepara os parâmetros do SVM
        svm_params = {
            'kernel': self.svm_params['kernel'],
            'C': self.svm_params['C'],
            'gamma': self.svm_params['gamma'],
            'degree': self.svm_params['degree'],
            'coef0': self.svm_params['coef0'],
            'class_weight': self.svm_params['class_weight'],
            'decision_function_shape': self.svm_params['decision_function_shape']
        }

        # Ajusta o parâmetro 'class_weight' se necessário
        if svm_params['class_weight'] == 'None' or svm_params['class_weight'] == '':
            svm_params['class_weight'] = None
        elif svm_params['class_weight'] == 'balanced':
            svm_params['class_weight'] = 'balanced'

        # Treina o classificador SVM usando os parâmetros definidos
        clf = SVC(**svm_params)
        clf.fit(X_train, y_train)

        # Faz predições no conjunto de teste
        y_pred = clf.predict(X_test)

        # Calcula as métricas
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        # Verifica o shape da matriz de confusão
        if cm.shape != (2, 2):
            cm_new = np.zeros((2, 2), dtype=int)
            for i, label in enumerate([0, 1]):
                if label in y_test or label in y_pred:
                    idx = np.where((y_test == label) | (y_pred == label))[0]
                    cm_new[i, :] = cm[i, :] if i < cm.shape[0] else [0, 0]
            cm = cm_new

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        return acc, sensitivity, specificity, cm

    # Método principal para a classificação com SVM
    def classificar_com_svm(self, retornar_metricas=False):
        # Verifica se o arquivo data.csv existe
        if not os.path.isfile('data.csv'):
            messagebox.showerror("Erro", "Arquivo 'data.csv' não encontrado. Por favor, gere o arquivo primeiro.")
            return

        # Carrega os dados do CSV
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

        # Executa a validação cruzada
        avg_accuracy, avg_sensitivity, avg_specificity, conf_matrices, _ = self.cross_validate_model(
            X, y_encoded, patient_numbers, self.train_and_evaluate_svm
        )

        if retornar_metricas:
            # Retorna as métricas e a matriz de confusão total
            total_conf_matrix = np.sum(conf_matrices, axis=0)
            return avg_accuracy, avg_sensitivity, avg_specificity, total_conf_matrix
        else:
            # Exibe os resultados
            self.display_classification_results(
                avg_accuracy, avg_sensitivity, avg_specificity, conf_matrices, le, model_name="SVM"
            )

    # Função para treinar e avaliar o MobileNet
    def train_and_evaluate_mobilenet(self, X_train, X_test, y_train, y_test):
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from sklearn.metrics import confusion_matrix, accuracy_score

        # Pré-processamento das imagens
        X_train = preprocess_input(X_train)
        X_test = preprocess_input(X_test)

        # Criação do modelo com MobileNet pré-treinado
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Descongela as camadas para fine-tuning, se especificado
        if self.mobilenet_params['fine_tune_layers'] > 0:
            for layer in base_model.layers[-self.mobilenet_params['fine_tune_layers']:]:
                layer.trainable = True
        else:
            for layer in base_model.layers:
                layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.mobilenet_params['dropout_rate'])(x)
        x = Dense(1024, activation=self.mobilenet_params['activation_function'])(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Configura o otimizador com a taxa de aprendizado especificada
        optimizer_name = self.mobilenet_params['optimizer']
        learning_rate = self.mobilenet_params['learning_rate']

        if optimizer_name.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.mobilenet_params.get('momentum', 0.0)
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        else:
            optimizer = tf.keras.optimizers.get(optimizer_name)
            optimizer.learning_rate = learning_rate  # Define a taxa de aprendizado

        # Compila o modelo usando os parâmetros definidos
        model.compile(
            optimizer=optimizer,
            loss=self.mobilenet_params['loss_function'],
            metrics=['accuracy']
        )

        # Define callbacks com o patience especificado
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.mobilenet_params['early_stopping_patience'],
                restore_best_weights=True
            )
        ]

        # Treinamento do modelo usando os parâmetros definidos
        history = model.fit(
            X_train, y_train,
            epochs=self.mobilenet_params['epochs'],
            batch_size=self.mobilenet_params['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        # Faz predições no conjunto de teste
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        # Calcula as métricas
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        # Verifica o shape da matriz de confusão
        if cm.shape != (2, 2):
            cm_new = np.zeros((2, 2), dtype=int)
            for i, label in enumerate([0, 1]):
                if label in y_test or label in y_pred:
                    idx = np.where((y_test == label) | (y_pred == label))[0]
                    cm_new[i, :] = cm[i, :] if i < cm.shape[0] else [0, 0]
            cm = cm_new

        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        return acc, sensitivity, specificity, cm, history

    # Método principal para a classificação com MobileNet
    def classificar_com_mobilenet(self, retornar_metricas=False):
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        from sklearn.preprocessing import LabelEncoder

        # Verifica se o arquivo data.csv existe
        if not os.path.isfile('data.csv'):
            messagebox.showerror("Erro", "Arquivo 'data.csv' não encontrado. Por favor, gere o arquivo primeiro.")
            return

        # Carrega os dados do CSV
        data = pd.read_csv('data.csv', delimiter=';')

        # Carrega as imagens e rótulos
        image_files = data['nome_arquivo'].values
        y = data['classe'].values

        # Codifica as labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Extrai os números dos pacientes dos nomes dos arquivos
        patient_numbers = self.extract_patient_numbers(data)

        # Diretório das imagens
        image_dir = "ROIS"  # Ajuste o diretório conforme necessário (onde estão as imagens ROI_*.png)

        # Carrega as imagens
        images = []
        for img_file in image_files:
            full_path = os.path.join(image_dir, img_file)
            if os.path.isfile(full_path):
                img = load_img(full_path, target_size=(224, 224))
                img_array = img_to_array(img)
                images.append(img_array)
            else:
                print(f"Arquivo {full_path} não encontrado.")
                messagebox.showerror("Erro", f"Arquivo {full_path} não encontrado.")
                return

        X = np.array(images)
        y_encoded = np.array(y_encoded)

        # Executa a validação cruzada
        avg_accuracy, avg_sensitivity, avg_specificity, conf_matrices, histories = self.cross_validate_model(
            X, y_encoded, patient_numbers, self.train_and_evaluate_mobilenet
        )

        if retornar_metricas:
            # Retorna as métricas e a matriz de confusão total
            total_conf_matrix = np.sum(conf_matrices, axis=0)
            return avg_accuracy, avg_sensitivity, avg_specificity, total_conf_matrix
        else:
            # Exibe os resultados
            self.display_classification_results(
                avg_accuracy, avg_sensitivity, avg_specificity, conf_matrices, le, model_name="MobileNet"
            )
            # Plota as curvas de aprendizado médias
            self.plot_learning_curves(histories)

    # Método para plotar as curvas de aprendizado
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

        # Plota as curvas de aprendizado
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, avg_train_acc, label='Acurácia de Treino')
        plt.plot(epochs_range, avg_val_acc, label='Acurácia de Validação')
        plt.legend(loc='lower right')
        plt.title('Acurácia Média de Treino e Validação por Época')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.tight_layout()
        plt.show()

    # Método para classificar e comparar os modelos
    def classificar_e_comparar(self):
        # Executa a classificação com SVM
        svm_results = self.classificar_com_svm(retornar_metricas=True)

        # Executa a classificação com MobileNet
        mobilenet_results = self.classificar_com_mobilenet(retornar_metricas=True)

        if svm_results and mobilenet_results:
            # Desempacota os resultados
            avg_accuracy_svm, avg_sensitivity_svm, avg_specificity_svm, conf_matrix_svm = svm_results
            avg_accuracy_mobilenet, avg_sensitivity_mobilenet, avg_specificity_mobilenet, conf_matrix_mobilenet = mobilenet_results

            # Exibe a tabela comparativa
            self.exibir_tabela_comparativa(
                avg_accuracy_svm, avg_sensitivity_svm, avg_specificity_svm, conf_matrix_svm,
                avg_accuracy_mobilenet, avg_sensitivity_mobilenet, avg_specificity_mobilenet, conf_matrix_mobilenet
            )

    # Método para exibir a tabela comparativa
    def exibir_tabela_comparativa(
        self,
        avg_accuracy_svm, avg_sensitivity_svm, avg_specificity_svm, conf_matrix_svm,
        avg_accuracy_mobilenet, avg_sensitivity_mobilenet, avg_specificity_mobilenet, conf_matrix_mobilenet
    ):
        # Cria uma janela para exibir os resultados
        result_window = Toplevel(self.root)
        result_window.title("Comparação de Classificadores")

        # Cria uma tabela usando o módulo ttk
        from tkinter import ttk

        # Dados para a tabela
        metrics = ["Acurácia Média", "Sensibilidade Média", "Especificidade Média"]
        svm_values = [f"{avg_accuracy_svm:.4f}", f"{avg_sensitivity_svm:.4f}", f"{avg_specificity_svm:.4f}"]
        mobilenet_values = [f"{avg_accuracy_mobilenet:.4f}", f"{avg_sensitivity_mobilenet:.4f}", f"{avg_specificity_mobilenet:.4f}"]

        # Configura a tabela
        tree = ttk.Treeview(result_window, columns=("Métrica", "SVM", "MobileNet"), show='headings')
        tree.heading("Métrica", text="Métrica")
        tree.heading("SVM", text="SVM")
        tree.heading("MobileNet", text="MobileNet")

        # Insere os dados na tabela
        for metric, svm_val, mobilenet_val in zip(metrics, svm_values, mobilenet_values):
            tree.insert("", "end", values=(metric, svm_val, mobilenet_val))

        tree.pack(pady=10)

        # Exibe as matrizes de confusão lado a lado
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matriz de Confusão - SVM')
        axes[0].set_xlabel('Predição')
        axes[0].set_ylabel('Verdadeiro')

        sns.heatmap(conf_matrix_mobilenet, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Matriz de Confusão - MobileNet')
        axes[1].set_xlabel('Predição')
        axes[1].set_ylabel('Verdadeiro')

        # Insere o gráfico na janela Tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Fecha a figura para liberar memória
        plt.close(fig)

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
        svm_kernel = StringVar(value=self.svm_params.get('kernel', 'linear'))
        Entry(svm_frame, textvariable=svm_kernel).grid(row=0, column=1, padx=5, pady=5)

        Label(svm_frame, text="C:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        svm_c = DoubleVar(value=self.svm_params.get('C', 1.0))
        Entry(svm_frame, textvariable=svm_c).grid(row=1, column=1, padx=5, pady=5)

        Label(svm_frame, text="Gamma:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        svm_gamma = StringVar(value=self.svm_params.get('gamma', 'scale'))
        Entry(svm_frame, textvariable=svm_gamma).grid(row=2, column=1, padx=5, pady=5)

        Label(svm_frame, text="Degree (para kernel 'poly'):").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        svm_degree = IntVar(value=self.svm_params.get('degree', 3))
        Entry(svm_frame, textvariable=svm_degree).grid(row=3, column=1, padx=5, pady=5)

        Label(svm_frame, text="Coef0 (para kernels 'poly' e 'sigmoid'):").grid(row=4, column=0, padx=5, pady=5, sticky='e')
        svm_coef0 = DoubleVar(value=self.svm_params.get('coef0', 0.0))
        Entry(svm_frame, textvariable=svm_coef0).grid(row=4, column=1, padx=5, pady=5)

        Label(svm_frame, text="Class Weight:").grid(row=5, column=0, padx=5, pady=5, sticky='e')
        svm_class_weight = StringVar(value=self.svm_params.get('class_weight', 'None'))
        Entry(svm_frame, textvariable=svm_class_weight).grid(row=5, column=1, padx=5, pady=5)

        Label(svm_frame, text="Decision Function Shape:").grid(row=6, column=0, padx=5, pady=5, sticky='e')
        svm_decision_function_shape = StringVar(value=self.svm_params.get('decision_function_shape', 'ovr'))
        Entry(svm_frame, textvariable=svm_decision_function_shape).grid(row=6, column=1, padx=5, pady=5)

        # --- Parâmetros do MobileNet ---
        Label(mobilenet_frame, text="Número de Épocas:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        mobilenet_epochs = IntVar(value=self.mobilenet_params.get('epochs', 5))
        Entry(mobilenet_frame, textvariable=mobilenet_epochs).grid(row=0, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Batch Size:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        mobilenet_batch_size = IntVar(value=self.mobilenet_params.get('batch_size', 16))
        Entry(mobilenet_frame, textvariable=mobilenet_batch_size).grid(row=1, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Otimizador:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        mobilenet_optimizer = StringVar(value=self.mobilenet_params.get('optimizer', 'adam'))
        Entry(mobilenet_frame, textvariable=mobilenet_optimizer).grid(row=2, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Learning Rate:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        mobilenet_learning_rate = DoubleVar(value=self.mobilenet_params.get('learning_rate', 0.001))
        Entry(mobilenet_frame, textvariable=mobilenet_learning_rate).grid(row=3, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Fine-tune Layers (Descongelar N camadas):").grid(row=4, column=0, padx=5, pady=5, sticky='e')
        mobilenet_fine_tune_layers = IntVar(value=self.mobilenet_params.get('fine_tune_layers', 0))
        Entry(mobilenet_frame, textvariable=mobilenet_fine_tune_layers).grid(row=4, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Early Stopping Patience:").grid(row=5, column=0, padx=5, pady=5, sticky='e')
        mobilenet_early_stopping_patience = IntVar(value=self.mobilenet_params.get('early_stopping_patience', 3))
        Entry(mobilenet_frame, textvariable=mobilenet_early_stopping_patience).grid(row=5, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Loss Function:").grid(row=6, column=0, padx=5, pady=5, sticky='e')
        mobilenet_loss_function = StringVar(value=self.mobilenet_params.get('loss_function', 'binary_crossentropy'))
        Entry(mobilenet_frame, textvariable=mobilenet_loss_function).grid(row=6, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Dropout Rate:").grid(row=7, column=0, padx=5, pady=5, sticky='e')
        mobilenet_dropout_rate = DoubleVar(value=self.mobilenet_params.get('dropout_rate', 0.0))
        Entry(mobilenet_frame, textvariable=mobilenet_dropout_rate).grid(row=7, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Activation Function:").grid(row=8, column=0, padx=5, pady=5, sticky='e')
        mobilenet_activation_function = StringVar(value=self.mobilenet_params.get('activation_function', 'relu'))
        Entry(mobilenet_frame, textvariable=mobilenet_activation_function).grid(row=8, column=1, padx=5, pady=5)

        Label(mobilenet_frame, text="Momentum (para SGD):").grid(row=9, column=0, padx=5, pady=5, sticky='e')
        mobilenet_momentum = DoubleVar(value=self.mobilenet_params.get('momentum', 0.0))
        Entry(mobilenet_frame, textvariable=mobilenet_momentum).grid(row=9, column=1, padx=5, pady=5)

        # Botão para salvar as configurações
        def salvar_parametros():
            # Atualiza os parâmetros do SVM
            self.svm_params['kernel'] = svm_kernel.get()
            self.svm_params['C'] = svm_c.get()
            self.svm_params['gamma'] = svm_gamma.get()
            self.svm_params['degree'] = svm_degree.get()
            self.svm_params['coef0'] = svm_coef0.get()
            class_weight_value = svm_class_weight.get()
            self.svm_params['class_weight'] = None if class_weight_value == 'None' else class_weight_value
            self.svm_params['decision_function_shape'] = svm_decision_function_shape.get()

            # Atualiza os parâmetros do MobileNet
            self.mobilenet_params['epochs'] = mobilenet_epochs.get()
            self.mobilenet_params['batch_size'] = mobilenet_batch_size.get()
            self.mobilenet_params['optimizer'] = mobilenet_optimizer.get()
            self.mobilenet_params['learning_rate'] = mobilenet_learning_rate.get()
            self.mobilenet_params['fine_tune_layers'] = mobilenet_fine_tune_layers.get()
            self.mobilenet_params['early_stopping_patience'] = mobilenet_early_stopping_patience.get()
            self.mobilenet_params['loss_function'] = mobilenet_loss_function.get()
            self.mobilenet_params['dropout_rate'] = mobilenet_dropout_rate.get()
            self.mobilenet_params['activation_function'] = mobilenet_activation_function.get()
            self.mobilenet_params['momentum'] = mobilenet_momentum.get()

            messagebox.showinfo("Parâmetros Salvos", "Os parâmetros foram atualizados com sucesso.")
            param_window.destroy()

        Button(param_window, text="Salvar", command=salvar_parametros).pack(pady=10)


# iniciar o aplicativo
root = Tk()
app = App(root)
app.mainloop()
