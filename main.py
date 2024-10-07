import os
from pathlib import Path
import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, Scale, HORIZONTAL, Menu, filedialog, simpledialog, messagebox, Toplevel, StringVar, IntVar, Radiobutton, Entry
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

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
        self.images = None
        self.total_patients = 0
        self.images_per_patient = 0
        self.current_patient = 0
        self.current_image = 0
        self.roi = None
        self.roi_zoom = None
        self.intensity_scale = 1
        self.zoom_factor = 1  # Controle de zoom inicial
        
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
        else:
            messagebox.showwarning("Aviso", "Nenhuma ROI foi selecionada. Por favor, selecione uma ROI primeiro.")

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

            # Salvar a ROI do fígado com o nome no formato 'ROI_nn_m'
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
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem foi carregada. Por favor, carregue uma imagem primeiro.")

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

            # Salvar a ROI do fígado com o nome no formato 'ROI_nn_m'
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
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem foi carregada. Por favor, carregue uma imagem primeiro.")

# Criar janela principal
root = Tk()
app = ImageViewer(root)
root.mainloop()
