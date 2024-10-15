import os
import csv
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, Menu, filedialog, simpledialog, messagebox, Toplevel, StringVar, Radiobutton, Entry
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops
import scipy.io
import pyfeats as pf

# Interface gráfica usando Tkinter
class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Imagens em Tons de Cinza com ROIs")
        self.root.geometry("800x600")
        
        # Variáveis para configurações do histograma
        self.bins = 256  # Padrão para o eixo X (bins)
        self.y_limit = 0.02  # Padrão para o eixo Y
        
        # Criando barra de menu
        self.menu_bar = Menu(root)
        root.config(menu=self.menu_bar)

        # Menu Arquivos
        self.arquivos_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Arquivos", menu=self.arquivos_menu)

        # Adicionar opções ao menu Arquivos
        self.arquivos_menu.add_command(label="Carregar banco de imagens (mat)", command=self.load_image_bank)
        self.arquivos_menu.add_command(label="Carregar imagem (png ou jpg)", command=self.load_image_file)
        self.arquivos_menu.add_command(label="Visualizar imagens", command=self.visualize_images)

        # Menu principal
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Opções", menu=self.file_menu)

        # Alterar o nome para "Carregar para Edição" e adicionar opções
        self.file_menu.add_command(label="Carregar para Edição", command=self.load_image)

        # Submenu para opções de ROI
        self.roi_menu = Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Opções de ROI", menu=self.roi_menu)

        # Opções dentro do submenu de ROI
        self.roi_menu.add_command(label="Selecionar ROI", command=self.select_roi)
        self.roi_menu.add_command(label="Mostrar Histograma da ROI", command=self.show_histogram)
        self.roi_menu.add_command(label="Gerar ROIs Manuais", command=self.generate_rois_manual)
        self.roi_menu.add_command(label="Gerar ROIs Manuais para Todo o Dataset", command=self.generate_rois_manual_dataset)
        self.roi_menu.add_command(label="Calcular GLCM e Descritores de Textura", command=self.calculate_glcm_texture)
        self.roi_menu.add_command(label="Calcular Descritor de Textura - SFM", command=self.calculate_texture_sfm)

        # Adiciona opção para mostrar histograma da imagem completa
        self.file_menu.add_command(label="Mostrar Histograma da Imagem", command=self.show_image_histogram)

        # Adiciona submenu de configurações
        self.settings_menu = Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Configurações", menu=self.settings_menu)

        # Opções dentro do submenu de configurações
        self.settings_menu.add_command(label="Ajustar Escala X (Bins)", command=self.adjust_bins)
        self.settings_menu.add_command(label="Ajustar Escala Y (Limite)", command=self.adjust_y_limit)
        self.settings_menu.add_command(label="Exibir Métricas da Imagem", command=self.show_image_metrics)

        # Exibir imagem e ROI
        self.label = Label(root)
        self.label.pack()

        self.zoom_slider = None  # Iniciando sem o slider
        self.img = None
        self.images = None
        self.total_patients = 0
        self.images_per_patient = 0
        self.current_patient = 0
        self.current_image = 0
        self.roi = None
        self.roi_zoom = None
        self.intensity_scale = 1
        self.zoom_factor = 1  # Controle de zoom inicial
        
        # Variáveis para o CSV
        self.csv_file = None
        self.csv_writer = None
        self.csv_file_path = "data.csv"  # Nome fixo para o arquivo CSV

        # Variável para controlar a interrupção do processo
        self.stop_processing = False

    def load_image_bank(self):
        # Abrir diálogo para selecionar arquivo .mat
        mat_file_path = filedialog.askopenfilename(title="Selecione o arquivo .mat", filetypes=[("MAT files", "*.mat")])
        if mat_file_path:
            # Carregar dados do arquivo .mat selecionado
            data = scipy.io.loadmat(mat_file_path)
            data_array = data['data']
            self.images = data_array['images']
            self.total_patients = len(self.images[0])
            self.images_per_patient = len(self.images[0][0])
            messagebox.showinfo("Sucesso", "Banco de imagens carregado com sucesso!")
            self.current_patient = 0
            self.current_image = 0
        else:
            messagebox.showwarning("Aviso", "Nenhum arquivo .mat foi selecionado.")

    def load_image_file(self):
        # Abrir diálogo para selecionar imagem
        image_file_path = filedialog.askopenfilename(title="Selecione a imagem", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if image_file_path:
            # Carregar imagem
            self.img = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
            if self.img is not None:
                self.show_image()
            else:
                messagebox.showerror("Erro", "Não foi possível carregar a imagem selecionada.")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem foi selecionada.")

    def visualize_images(self):
        if self.images is not None:
            # Criar janela de visualização de imagens
            self.visualization_window = Toplevel(self.root)
            self.visualization_window.title("Visualizar Imagens")
            
            # Label para exibir a imagem
            self.image_label = Label(self.visualization_window)
            self.image_label.pack()
            
            # Label para exibir o número do paciente e da foto
            self.info_label = Label(self.visualization_window, text="")
            self.info_label.pack()
            
            # Botões de navegação
            prev_button = Button(self.visualization_window, text="Anterior", command=self.show_prev_image)
            prev_button.pack(side='left')
            next_button = Button(self.visualization_window, text="Próxima", command=self.show_next_image)
            next_button.pack(side='right')
            
            # Exibir a primeira imagem
            self.current_patient = 0
            self.current_image = 0
            self.show_current_image_in_visualization()
        else:
            messagebox.showwarning("Aviso", "Nenhum banco de imagens carregado. Por favor, carregue um banco de imagens primeiro.")

    def show_current_image_in_visualization(self):
        # Carregar imagem do paciente e imagem atual
        self.img = self.images[0][self.current_patient][self.current_image]
        if self.img is not None:
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img))
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            
            # Atualizar o texto com o número do paciente e da foto
            info_text = f"Paciente: {self.current_patient}/{self.total_patients - 1}, Imagem: {self.current_image}/{self.images_per_patient - 1}"
            self.info_label.config(text=info_text)
        else:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem.")

    def show_next_image(self):
        self.current_image = (self.current_image + 1) % self.images_per_patient
        if self.current_image == 0:
            self.current_patient = (self.current_patient + 1) % self.total_patients
        self.show_current_image_in_visualization()

    def show_prev_image(self):
        if self.current_image == 0:
            self.current_image = self.images_per_patient - 1
            self.current_patient = (self.current_patient - 1) % self.total_patients
        else:
            self.current_image -=1
        self.show_current_image_in_visualization()

    def load_image(self):
        if self.images is not None:
            # Criar janela para escolher entre 'Aleatória' e 'Específica'
            choice_window = Toplevel(self.root)
            choice_window.title("Carregar Imagem")
            choice_window.grab_set()  # Focar nesta janela

            choice_var = StringVar(value="aleatoria")

            Label(choice_window, text="Escolha uma opção:").pack(pady=10)

            Radiobutton(choice_window, text="Aleatória", variable=choice_var, value="aleatoria").pack(anchor='w')
            Radiobutton(choice_window, text="Específica", variable=choice_var, value="especifica").pack(anchor='w')

            def confirm_choice():
                choice = choice_var.get()
                choice_window.destroy()
                if choice == 'aleatoria':
                    # Carregar imagem aleatória
                    self.current_patient = np.random.randint(0, self.total_patients)
                    self.current_image = np.random.randint(0, self.images_per_patient)
                    self.img = self.images[0][self.current_patient][self.current_image]
                    self.show_image()
                elif choice == 'especifica':
                    # Abrir janela para inserir números
                    self.ask_patient_image_numbers()
                else:
                    messagebox.showwarning("Aviso", "Opção inválida.")

            Button(choice_window, text="OK", command=confirm_choice).pack(pady=10)
        else:
            messagebox.showwarning("Aviso", "Nenhum banco de imagens carregado. Por favor, carregue um banco de imagens primeiro.")

    def ask_patient_image_numbers(self):
        # Janela para inserir o número do paciente e da imagem
        input_window = Toplevel(self.root)
        input_window.title("Especificar Imagem")
        input_window.grab_set()  # Focar nesta janela

        Label(input_window, text=f"Digite o número do paciente (0-{self.total_patients - 1}):").pack(pady=5)
        patient_entry = Entry(input_window)
        patient_entry.pack(pady=5)

        Label(input_window, text=f"Digite o número da imagem (0-{self.images_per_patient - 1}):").pack(pady=5)
        image_entry = Entry(input_window)
        image_entry.pack(pady=5)

        def load_specific_image():
            try:
                patient_num = int(patient_entry.get())
                image_num = int(image_entry.get())
                if 0 <= patient_num <= self.total_patients - 1 and 0 <= image_num <= self.images_per_patient - 1:
                    self.current_patient = patient_num
                    self.current_image = image_num
                    self.img = self.images[0][self.current_patient][self.current_image]
                    self.show_image()
                    input_window.destroy()
                else:
                    messagebox.showerror("Erro", "Números fora do intervalo válido.")
            except ValueError:
                messagebox.showerror("Erro", "Por favor, insira números inteiros válidos.")

        Button(input_window, text="Carregar Imagem", command=load_specific_image).pack(pady=10)

    def show_image(self):
        if self.img is not None:
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img))
            self.label.config(image=img_tk)
            self.label.image = img_tk
        else:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem.")

    def select_roi(self):
        # Se uma imagem foi carregada, permite a seleção de uma ROI
        if self.img is not None:
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)  # Converter para BGR para exibição com OpenCV
            self.roi = cv2.selectROI("Selecione a ROI", img_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Selecione a ROI")

            # Cortar a ROI da imagem original
            x, y, w, h = self.roi
            self.roi_zoom = self.img[y:y+h, x:x+w]

            # Criar botão "Visualizar ROI" após seleção
            if not hasattr(self, 'view_roi_button'):
                self.view_roi_button = Button(self.root, text="Visualizar ROI", command=self.show_roi)
                self.view_roi_button.pack()
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem foi carregada. Por favor, carregue uma imagem primeiro.")

    def show_roi(self):
        if self.roi_zoom is not None:
            # Exibir a ROI usando matplotlib
            plt.figure(figsize=(5, 5))
            plt.imshow(self.roi_zoom, cmap='gray')
            plt.title("ROI Selecionada")
            plt.axis('on')  # Exibir eixos, ou use 'off' para ocultá-los
            plt.show()

    def update_zoom(self, value):
        self.zoom_factor = float(value)
        self.show_roi()

    def show_histogram(self):
        if self.roi_zoom is not None:
            plt.figure()
            # Exibir histograma da ROI com as configurações ajustadas
            plt.hist(self.roi_zoom.ravel(), bins=self.bins, range=(0, 255), density=True, color='gray', edgecolor='black')
            plt.title("Histograma da ROI")
            plt.xlabel("Níveis de Cinza")
            plt.ylabel("Densidade")
            plt.ylim(0, self.y_limit)  # Limite ajustável do eixo Y
            plt.show()
        else:
            messagebox.showwarning("Aviso", "Nenhuma ROI foi selecionada. Por favor, selecione uma ROI primeiro.")

    def calculate_glcm_texture(self):
        if self.roi_zoom is not None:
            # Utiliza a ROI selecionada para calcular os descritores
            entropies, homogeneities = self.calculate_glcm_descriptors(self.roi_zoom)
            result_text = "Descritores de Textura (GLCM) Radial:\n\n"
            distances = [1, 2, 4, 8]
            for i, d in enumerate(distances):
                result_text += (f"Distância {d} pixels:\n"
                                f"Entropia: {entropies[i]:.4f}\n"
                                f"Homogeneidade: {homogeneities[i]:.4f}\n\n")
            messagebox.showinfo("Descritores de Textura (GLCM)", result_text)
        else:
            messagebox.showwarning("Aviso", "Nenhuma ROI foi selecionada. Por favor, selecione uma ROI primeiro.")

    def calculate_glcm_descriptors(self, roi):
        roi_norm = (roi / np.max(roi) * 255).astype(np.uint8)
        distances = [1, 2, 4, 8]
        num_angles = 16
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        
        entropies = []
        homogeneities = []

        for d in distances:
            glcm = graycomatrix(roi_norm, distances=[d], angles=angles, levels=256, symmetric=True, normed=True)
            glcm_radial = np.sum(glcm, axis=3)
            glcm_radial = glcm_radial / np.sum(glcm_radial)

            glcm_nonzero = glcm_radial[glcm_radial > 0]
            entropy_val = -np.sum(glcm_nonzero * np.log2(glcm_nonzero))
            entropies.append(entropy_val)

            homogeneity_val = np.sum(glcm_radial / (1 + np.abs(np.arange(256)[:, None] - np.arange(256))))
            homogeneities.append(homogeneity_val)

        return entropies, homogeneities

    def show_glcm_matrix(self, glcm, distance):
        plt.figure(figsize=(8, 6))
        plt.imshow(glcm[:, :, 0], cmap='gray')  # Ajustado para 3 dimensões
        plt.colorbar()
        plt.title(f'Matriz de Coocorrência - Distância {distance}')
        plt.xlabel('Níveis de Cinza')
        plt.ylabel('Níveis de Cinza')
        plt.show()

    def calculate_texture_sfm(self):
        if self.roi_zoom is not None:
            # Utiliza a ROI selecionada para calcular os descritores
            features, labels = pf.sfm_features(f=self.roi_zoom, mask=None, Lr=4, Lc=4)
            coarseness = features[0]
            contrast = features[1]
            periodicity = features[2]
            roughness = features[3]

            result_text = (f"Descritores de Textura (SFM):\n\n"
                           f"Coarseness: {coarseness:.4f}\n"
                           f"Contrast: {contrast:.4f}\n"
                           f"Periodicity: {periodicity:.4f}\n"
                           f"Roughness: {roughness:.4f}")
            messagebox.showinfo("Descritores de Textura (SFM)", result_text)
        else:
            messagebox.showwarning("Aviso", "Nenhuma ROI foi selecionada. Por favor, selecione uma ROI primeiro.")

    def calculate_coarseness(self, roi):
        # Exemplo de cálculo de coarseness
        return np.var(roi)

    def calculate_contrast(self, roi):
        # Exemplo de cálculo de contrast
        return np.mean(np.abs(np.diff(roi, axis=0))) + np.mean(np.abs(np.diff(roi, axis=1)))

    def calculate_periodicity(self, roi):
        # Exemplo de cálculo de periodicity
        return np.mean(np.cos(roi))

    def calculate_roughness(self, roi):
        # Exemplo de cálculo de roughness
        return np.std(roi)

    def show_image_histogram(self):
        if self.img is not None:
            plt.figure()
            # Exibir histograma da imagem completa com as configurações ajustadas
            plt.hist(self.img.ravel(), bins=self.bins, range=(0, 255), density=True, color='gray', edgecolor='black')
            plt.title("Histograma da Imagem Completa")
            plt.xlabel("Níveis de Cinza")
            plt.ylabel("Densidade")
            plt.ylim(0, self.y_limit)  # Limite ajustável do eixo Y
            plt.show()
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem foi carregada. Por favor, carregue uma imagem primeiro.")

    def adjust_bins(self):
        # Ajustar a escala do eixo X (bins) através de uma caixa de diálogo
        new_bins = simpledialog.askinteger("Escala X (Bins)", "Digite o número de bins:", minvalue=1, maxvalue=512)
        if new_bins:
            self.bins = new_bins

    def adjust_y_limit(self):
        # Ajustar o limite do eixo Y através de uma caixa de diálogo
        new_y_limit = simpledialog.askfloat("Escala Y (Limite)", "Digite o limite superior do eixo Y:", minvalue=0.001, maxvalue=1.0)
        if new_y_limit:
            self.y_limit = new_y_limit

    def show_image_metrics(self):
        if self.img is not None:
            # Calcular as métricas: média, mediana e desvio padrão
            mean_val = np.mean(self.img)
            median_val = np.median(self.img)
            std_dev = np.std(self.img)
            
            # Exibir os resultados em uma nova janela de diálogo
            metrics_text = f"Média: {mean_val:.2f}\nMediana: {median_val:.2f}\nDesvio Padrão: {std_dev:.2f}"
            simpledialog.messagebox.showinfo("Métricas da Imagem", metrics_text)
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem foi carregada. Por favor, carregue uma imagem primeiro.")

    def generate_rois_manual(self):
        # Geração manual das ROIs para o fígado e córtex renal com tamanho fixo de 28x28 pixels
        if self.img is not None:
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

            # Selecionar manualmente a posição da ROI do fígado
            liver_point = self.get_click_point(img_bgr, "Clique na posição da ROI do Fígado")
            if liver_point is None:
                messagebox.showwarning("Aviso", "Nenhum ponto foi selecionado para o fígado.")
                return
            x_liver, y_liver = liver_point

            # Garantir que a ROI esteja dentro dos limites da imagem
            x_liver_start = max(0, x_liver - 14)
            y_liver_start = max(0, y_liver - 14)
            x_liver_end = x_liver_start + 28
            y_liver_end = y_liver_start + 28

            # Ajustar se a ROI ultrapassar as dimensões da imagem
            if x_liver_end > self.img.shape[1]:
                x_liver_end = self.img.shape[1]
                x_liver_start = x_liver_end - 28
            if y_liver_end > self.img.shape[0]:
                y_liver_end = self.img.shape[0]
                y_liver_start = y_liver_end - 28

            liver_roi = self.img[y_liver_start:y_liver_end, x_liver_start:x_liver_end]

            # Selecionar manualmente a posição da ROI do rim
            kidney_point = self.get_click_point(img_bgr, "Clique na posição da ROI do Rim")
            if kidney_point is None:
                messagebox.showwarning("Aviso", "Nenhum ponto foi selecionado para o rim.")
                return
            x_kidney, y_kidney = kidney_point

            x_kidney_start = max(0, x_kidney - 14)
            y_kidney_start = max(0, y_kidney - 14)
            x_kidney_end = x_kidney_start + 28
            y_kidney_end = y_kidney_start + 28

            if x_kidney_end > self.img.shape[1]:
                x_kidney_end = self.img.shape[1]
                x_kidney_start = x_kidney_end - 28
            if y_kidney_end > self.img.shape[0]:
                y_kidney_end = self.img.shape[0]
                y_kidney_start = y_kidney_end - 28

            kidney_roi = self.img[y_kidney_start:y_kidney_end, x_kidney_start:x_kidney_end]

            # Calcular o índice hepatorenal (HI)
            mean_liver = np.mean(liver_roi)
            mean_kidney = np.mean(kidney_roi)
            hi = mean_liver / mean_kidney if mean_kidney != 0 else 1

            # Normalizar a ROI do fígado
            liver_roi_adjusted = liver_roi * hi
            liver_roi_adjusted = np.round(liver_roi_adjusted).astype(np.uint8)

            # Salvar a ROI do fígado com o nome no formato 'ROI_nn_m'
            roi_filename = f"ROI_{self.current_patient:02d}_{self.current_image}.png"
            cv2.imwrite(roi_filename, liver_roi_adjusted)
            messagebox.showinfo("Sucesso", f"ROI do fígado salva como {roi_filename}")

            # Exibir as ROIs
            img_copy = img_bgr.copy()
            cv2.rectangle(img_copy, (x_liver_start, y_liver_start), (x_liver_end, y_liver_end), (0, 255, 0), 2)  # Verde para o fígado
            cv2.rectangle(img_copy, (x_kidney_start, y_kidney_start), (x_kidney_end, y_kidney_end), (255, 0, 0), 2)  # Azul para o rim

            img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("ROIs Manuais: Fígado (verde), Rim (azul)")
            plt.show()
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem foi carregada. Por favor, carregue uma imagem primeiro.")

    def generate_rois_manual_dataset(self):
        if self.images is not None:
            self.ask_starting_patient_image_numbers()
        else:
            messagebox.showwarning("Aviso", "Nenhum banco de imagens carregado. Por favor, carregue um banco de imagens primeiro.")

    def ask_starting_patient_image_numbers(self):
        # Janela para inserir o número do paciente e da imagem
        input_window = Toplevel(self.root)
        input_window.title("Especificar Ponto de Início")
        input_window.grab_set()  # Focar nesta janela

        Label(input_window, text=f"Digite o número do paciente inicial (0-{self.total_patients - 1}):").pack(pady=5)
        patient_entry = Entry(input_window)
        patient_entry.pack(pady=5)

        Label(input_window, text=f"Digite o número da imagem inicial (0-{self.images_per_patient - 1}):").pack(pady=5)
        image_entry = Entry(input_window)
        image_entry.pack(pady=5)

        def confirm_starting_point():
            try:
                patient_num = int(patient_entry.get())
                image_num = int(image_entry.get())
                if 0 <= patient_num <= self.total_patients - 1 and 0 <= image_num <= self.images_per_patient - 1:
                    self.current_patient = patient_num
                    self.current_image = image_num
                    input_window.destroy()
                    # Abrir o arquivo CSV para escrita ou append
                    file_exists = os.path.isfile(self.csv_file_path)
                    if file_exists:
                        self.csv_file = open(self.csv_file_path, 'a', newline='', encoding='utf-8')
                        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=['nome_arquivo', 'classe', 'figado_x', 'figado_y', 'rim_x', 'rim_y', 'HI',
                                                                                   'entropia', 'homogeneidade', 'coarseness', 'contrast', 'periodicity', 'roughness'], delimiter=';')
                    else:
                        self.csv_file = open(self.csv_file_path, 'w', newline='', encoding='utf-8')
                        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=['nome_arquivo', 'classe', 'figado_x', 'figado_y', 'rim_x', 'rim_y', 'HI',
                                                                                   'entropia', 'homogeneidade', 'coarseness', 'contrast', 'periodicity', 'roughness'], delimiter=';')
                        self.csv_writer.writeheader()
                    # Iniciar processamento
                    self.process_next_image_manual_roi()
                else:
                    messagebox.showerror("Erro", "Números fora do intervalo válido.")
            except ValueError:
                messagebox.showerror("Erro", "Por favor, insira números inteiros válidos.")

        Button(input_window, text="Iniciar", command=confirm_starting_point).pack(pady=10)

    def process_next_image_manual_roi(self):
        # Verificar se o usuário optou por parar o processamento
        if self.stop_processing:
            messagebox.showinfo("Processo Interrompido", "O processamento foi interrompido pelo usuário.")
            # Fechar o arquivo CSV
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            self.stop_processing = False  # Resetar para o próximo uso
            return

        # Verificar se chegamos ao final do conjunto de dados
        if self.current_patient >= self.total_patients:
            messagebox.showinfo("Concluído", "Processamento de todas as imagens concluído.")
            # Fechar o arquivo CSV
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
            return
        if self.current_image >= self.images_per_patient:
            # Mover para o próximo paciente
            self.current_patient += 1
            self.current_image = 0
            if self.current_patient >= self.total_patients:
                messagebox.showinfo("Concluído", "Processamento de todas as imagens concluído.")
                # Fechar o arquivo CSV
                if self.csv_file:
                    self.csv_file.close()
                    self.csv_file = None
                return
        # Carregar a imagem
        self.img = self.images[0][self.current_patient][self.current_image]
        self.show_image()
        # Proceder para gerar ROIs manuais para esta imagem
        self.generate_rois_manual_for_image()

    def generate_rois_manual_for_image(self):
        # Geração manual das ROIs para o fígado e córtex renal com tamanho fixo de 28x28 pixels
        if self.img is not None:
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

            # Selecionar manualmente a posição da ROI do fígado
            liver_point = self.get_click_point(img_bgr, f"Paciente {self.current_patient}, Imagem {self.current_image}: Clique na posição da ROI do Fígado")
            if liver_point is None:
                messagebox.showinfo("Processo Interrompido", "O processamento foi interrompido pelo usuário.")
                self.stop_processing = True
                self.process_next_image_manual_roi()
                return
            x_liver, y_liver = liver_point

            # Garantir que a ROI esteja dentro dos limites da imagem
            x_liver_start = max(0, x_liver - 14)
            y_liver_start = max(0, y_liver - 14)
            x_liver_end = x_liver_start + 28
            y_liver_end = y_liver_start + 28

            # Ajustar se a ROI ultrapassar as dimensões da imagem
            if x_liver_end > self.img.shape[1]:
                x_liver_end = self.img.shape[1]
                x_liver_start = x_liver_end - 28
            if y_liver_end > self.img.shape[0]:
                y_liver_end = self.img.shape[0]
                y_liver_start = y_liver_end - 28

            liver_roi = self.img[y_liver_start:y_liver_end, x_liver_start:x_liver_end]

            # Selecionar manualmente a posição da ROI do rim
            kidney_point = self.get_click_point(img_bgr, f"Paciente {self.current_patient}, Imagem {self.current_image}: Clique na posição da ROI do Rim")
            if kidney_point is None:
                messagebox.showinfo("Processo Interrompido", "O processamento foi interrompido pelo usuário.")
                self.stop_processing = True
                self.process_next_image_manual_roi()
                return
            x_kidney, y_kidney = kidney_point

            x_kidney_start = max(0, x_kidney - 14)
            y_kidney_start = max(0, y_kidney - 14)
            x_kidney_end = x_kidney_start + 28
            y_kidney_end = y_kidney_start + 28

            if x_kidney_end > self.img.shape[1]:
                x_kidney_end = self.img.shape[1]
                x_kidney_start = x_kidney_end - 28
            if y_kidney_end > self.img.shape[0]:
                y_kidney_end = self.img.shape[0]
                y_kidney_start = y_kidney_end - 28

            kidney_roi = self.img[y_kidney_start:y_kidney_end, x_kidney_start:x_kidney_end]

            # Calcular o índice hepatorenal (HI)
            mean_liver = np.mean(liver_roi)
            mean_kidney = np.mean(kidney_roi)
            hi = mean_liver / mean_kidney if mean_kidney != 0 else 1

            # Normalizar a ROI do fígado
            liver_roi_adjusted = liver_roi * hi
            liver_roi_adjusted = np.round(liver_roi_adjusted).astype(np.uint8)

            # Salvar a ROI do fígado com o nome no formato 'ROI_nn_m'
            roi_filename = f"ROI_{self.current_patient:02d}_{self.current_image}.png"
            cv2.imwrite(roi_filename, liver_roi_adjusted)
            # Opcionalmente, você pode exibir uma mensagem de sucesso
            messagebox.showinfo("Sucesso", f"ROI do fígado salva como {roi_filename}")

            # Exibir as ROIs
            img_copy = img_bgr.copy()
            cv2.rectangle(img_copy, (x_liver_start, y_liver_start), (x_liver_end, y_liver_end), (0, 255, 0), 2)  # Verde para o fígado
            cv2.rectangle(img_copy, (x_kidney_start, y_kidney_start), (x_kidney_end, y_kidney_end), (255, 0, 0), 2)  # Azul para o rim

            img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(f"ROIs Manuais: Fígado (verde), Rim (azul) - Paciente {self.current_patient}, Imagem {self.current_image}")
            plt.show()

            # Cálculo dos descritores de textura usando as funções existentes
            entropies, homogeneities = self.calculate_glcm_descriptors(liver_roi_adjusted)
            features, labels = pf.sfm_features(f=liver_roi_adjusted, mask=None, Lr=4, Lc=4)
            coarseness = features[0]
            contrast = features[1]
            periodicity = features[2]
            roughness = features[3]

            # Determinar a classe do paciente
            class_label = 'Saudavel' if self.current_patient <= 16 else 'Esteatose'

            # Remover vírgulas e caracteres especiais dos dados
            roi_filename_clean = re.sub(r'[^\w\-_\. ]', '_', roi_filename)
            class_label_clean = re.sub(r'[^\w\-_\. ]', '_', class_label)

            # Preparar os valores de entropia e homogeneidade (usando a distância 1 como exemplo)
            entropy = entropies[0]
            homogeneity = homogeneities[0]

            # Armazenar os dados em um dicionário
            data_row = {
                'nome_arquivo': roi_filename_clean,
                'classe': class_label_clean,
                'figado_x': x_liver_start,
                'figado_y': y_liver_start,
                'rim_x': x_kidney_start,
                'rim_y': y_kidney_start,
                'HI': f"{hi:.4f}",
                'entropia': f"{entropy:.4f}",
                'homogeneidade': f"{homogeneity:.4f}",
                'coarseness': f"{coarseness:.4f}",
                'contrast': f"{contrast:.4f}",
                'periodicity': f"{periodicity:.4f}",
                'roughness': f"{roughness:.4f}"
            }

            # Escrever a linha no arquivo CSV
            if self.csv_writer:
                self.csv_writer.writerow(data_row)
                self.csv_file.flush()  # Garantir que os dados sejam gravados imediatamente

            # Avançar para a próxima imagem
            self.current_image += 1
            self.process_next_image_manual_roi()
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem foi carregada. Por favor, carregue uma imagem primeiro.")

    def get_click_point(self, img, window_title):
        click_point = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_point.append((x, y))
                cv2.destroyWindow(window_title)

        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, mouse_callback)
        cv2.imshow(window_title, img)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Tecla Esc
                self.stop_processing = True
                cv2.destroyWindow(window_title)
                break
            if len(click_point) > 0:
                break

        if self.stop_processing:
            self.stop_processing = False  # Resetar para futuras chamadas
            return None
        elif click_point:
            return click_point[0]
        else:
            return None

# Criar janela principal
root = Tk()
app = ImageViewer(root)
root.mainloop()
