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

# iniciar o aplicativo
root = Tk()
app = App(root)
app.mainloop()
