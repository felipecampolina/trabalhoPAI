import os
from pathlib import Path
import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, Scale, HORIZONTAL, Menu, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

# Definir nome do diretório
path_input_dir = Path("../trabalhoPAI")
path_data = path_input_dir / "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"

# Carregar dados
data = scipy.io.loadmat(path_data)
data_array = data['data']
images = data_array['images']

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

        # Menu principal
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Opções", menu=self.file_menu)

        # Adiciona opção para carregar imagem
        self.file_menu.add_command(label="Carregar Imagem", command=self.load_image)

        # Submenu para opções de ROI
        self.roi_menu = Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Opções de ROI", menu=self.roi_menu)

        # Opções dentro do submenu de ROI
        self.roi_menu.add_command(label="Selecionar ROI", command=self.select_roi)
        self.roi_menu.add_command(label="Mostrar Histograma da ROI", command=self.show_histogram)
        self.roi_menu.add_command(label="Gerar ROIs Automáticas", command=self.generate_rois_automatic)
        self.roi_menu.add_command(label="Gerar ROIs Manualmente", command=self.generate_rois_manual)
        self.roi_menu.add_command(label="Calcular GLCM e Descritores de Textura", command=self.calculate_glcm_texture)
        self.roi_menu.add_command(label="Calcular Descritor De Textura - SFM ", command=self.calculate_texture_sfm)

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
        self.roi = None
        self.roi_zoom = None
        self.intensity_scale = 1
        self.current_patient = 0
        self.current_image = 0
        self.zoom_factor = 1  # Controle de zoom inicial
        
    def load_image(self):
        # Alternar entre as imagens, variando os pacientes (n) e as imagens (m)
        self.current_patient = (self.current_patient + 1) % 55  # Existem 55 pacientes (n = 0 a 54)
        self.current_image = (self.current_image + 1) % 10  # Cada paciente tem 10 imagens (m = 0 a 9)
        
        # Carregar imagem do paciente e imagem atual
        self.img = images[0][self.current_patient][self.current_image]
        
        # Exibir a imagem carregada
        self.show_image()

    def show_image(self):
        if self.img is not None:
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.img))
            self.label.config(image=img_tk)
            self.label.image = img_tk

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

    def calculate_glcm_texture(self):
        if self.roi_zoom is not None:
            roi_norm = (self.roi_zoom / np.max(self.roi_zoom) * 255).astype(np.uint8)
            distances = [1, 2, 4, 8]
            angles = [0]
            
            entropies = []
            homogeneities = []

            for d in distances:
                glcm = graycomatrix(roi_norm, distances=[d], angles=angles, levels=256, symmetric=True, normed=True)

                # Exibir a matriz de coocorrência para a distância atual
                self.show_glcm_matrix(glcm, d)

                entropy_val = shannon_entropy(glcm)
                entropies.append(entropy_val)

                homogeneity_val = graycoprops(glcm, 'homogeneity')[0, 0]
                homogeneities.append(homogeneity_val)

            result_text = "Descritores de Textura (GLCM):\n\n"
            for i, d in enumerate(distances):
                result_text += (f"Distância {d} pixels:\n"
                                f"Entropia: {entropies[i]:.4f}\n"
                                f"Homogeneidade: {homogeneities[i]:.4f}\n\n")

            messagebox.showinfo("Descritores de Textura (GLCM)", result_text)


    def show_glcm_matrix(self, glcm, distance):
        plt.figure(figsize=(8, 6))
        plt.imshow(glcm[:, :, 0, 0], cmap='gray')
        plt.colorbar()
        plt.title(f'Matriz de Coocorrência - Distância {distance}')
        plt.xlabel('Níveis de Cinza')
        plt.ylabel('Níveis de Cinza')
        plt.show()

    def calculate_texture_sfm(self):
        if self.roi_zoom is not None:
            roi_norm = (self.roi_zoom / np.max(self.roi_zoom) * 255).astype(np.uint8)

            # Descritores SFM
            coarseness = self.calculate_coarseness(roi_norm)
            contrast = self.calculate_contrast(roi_norm)
            periodicity = self.calculate_periodicity(roi_norm)
            roughness = self.calculate_roughness(roi_norm)

            result_text = (f"Descritores de Textura (SFM):\n\n"
                           f"Coarseness: {coarseness:.4f}\n"
                           f"Contrast: {contrast:.4f}\n"
                           f"Periodicity: {periodicity:.4f}\n"
                           f"Roughness: {roughness:.4f}")

            messagebox.showinfo("Descritores de Textura (SFM)", result_text)

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

    def generate_rois_automatic(self):
        # Geração automática das ROIs para o fígado e córtex renal
        if self.img is not None:
            # Coordenadas ajustadas para o fígado
            x_liver, y_liver = 220, 120
            liver_roi = self.img[y_liver:y_liver+28, x_liver:x_liver+28]

            # Coordenadas ajustadas para o córtex renal
            x_kidney, y_kidney = 250, 200
            kidney_roi = self.img[y_kidney:y_kidney+28, x_kidney:x_kidney+28]

            # Calcular o índice hepatorenal (HI)
            mean_liver = np.mean(liver_roi)
            mean_kidney = np.mean(kidney_roi)
            hi = mean_liver / mean_kidney if mean_kidney != 0 else 1

            # Normalizar a ROI do fígado
            liver_roi_adjusted = liver_roi * hi
            liver_roi_adjusted = np.round(liver_roi_adjusted).astype(np.uint8)

            # Salvar a ROI do fígado
            roi_filename = f"ROI_{self.current_patient:02d}_{self.current_image}.png"
            cv2.imwrite(roi_filename, liver_roi_adjusted)
            messagebox.showinfo("Sucesso", f"ROI do fígado salva como {roi_filename}")

            # Exibir as ROIs
            img_copy = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img_copy, (x_liver, y_liver), (x_liver+28, y_liver+28), (0, 255, 0), 2)  # Verde para o fígado
            cv2.rectangle(img_copy, (x_kidney, y_kidney), (x_kidney+28, y_kidney+28), (255, 0, 0), 2)  # Azul para o rim

            img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("ROIs Automáticas: Fígado (verde), Rim (azul)")
            plt.show()

    def generate_rois_manual(self):
        # Geração manual das ROIs para o fígado e córtex renal com tamanho fixo de 28x28 pixels
        if self.img is not None:
            # Selecionar manualmente a posição da ROI do fígado
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            roi_liver = cv2.selectROI("Selecione a posição da ROI do Fígado", img_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Selecione a posição da ROI do Fígado")

            x_liver, y_liver = int(roi_liver[0]), int(roi_liver[1])
            liver_roi = self.img[y_liver:y_liver+28, x_liver:x_liver+28]

            # Selecionar manualmente a posição da ROI do rim
            roi_kidney = cv2.selectROI("Selecione a posição da ROI do Rim", img_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Selecione a posição da ROI do Rim")

            x_kidney, y_kidney = int(roi_kidney[0]), int(roi_kidney[1])
            kidney_roi = self.img[y_kidney:y_kidney+28, x_kidney:x_kidney+28]

            # Calcular o índice hepatorenal (HI)
            mean_liver = np.mean(liver_roi)
            mean_kidney = np.mean(kidney_roi)
            hi = mean_liver / mean_kidney if mean_kidney != 0 else 1

            # Normalizar a ROI do fígado
            liver_roi_adjusted = liver_roi * hi
            liver_roi_adjusted = np.round(liver_roi_adjusted).astype(np.uint8)

            # Salvar a ROI do fígado
            roi_filename = f"ROI_{self.current_patient:02d}_{self.current_image}.png"
            cv2.imwrite(roi_filename, liver_roi_adjusted)
            messagebox.showinfo("Sucesso", f"ROI do fígado salva como {roi_filename}")

            # Exibir as ROIs
            img_copy = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img_copy, (x_liver, y_liver), (x_liver+28, y_liver+28), (0, 255, 0), 2)  # Verde para o fígado
            cv2.rectangle(img_copy, (x_kidney, y_kidney), (x_kidney+28, y_kidney+28), (255, 0, 0), 2)  # Azul para o rim

            img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title("ROIs Manuais: Fígado (verde), Rim (azul)")
            plt.show()

# Criar janela principal
root = Tk()
app = ImageViewer(root)
root.mainloop()
