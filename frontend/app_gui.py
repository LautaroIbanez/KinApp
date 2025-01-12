from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from backend.video_processor import VideoProcessor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Detection App")
        self.style = ttk.Style("superhero")  # Tema de ttkbootstrap

        # Inicializar componentes
        self.processor = VideoProcessor()

        # Variables del video
        self.cap = None
        self.video_path = None
        self.is_paused = False
        self.processed_frames = []
        self.selected_metrics = {
            "right_knee_angle": tk.BooleanVar(value=True),
            "left_knee_angle": tk.BooleanVar(value=True),
            "right_shoulder_angle": tk.BooleanVar(value=True),
            "left_shoulder_angle": tk.BooleanVar(value=True),
        }
        self.mode = tk.StringVar(value="relative")  # "relative" o "fixed"
        self.plane = tk.StringVar(value="horizontal")  # "horizontal" o "vertical"

        # Velocidad de reproducción
        self.play_speed = tk.DoubleVar(value=1.0)

        # Variables para gráficos
        self.graph_data = {
            "right_knee_angle": [],
            "left_knee_angle": [],
            "right_shoulder_angle": [],
            "left_shoulder_angle": []
        }

        # Crear la interfaz gráfica
        self.create_widgets()

    def create_widgets(self):
        """
        Crear los elementos de la GUI.
        """
        # Frame superior (controles)
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=X, pady=5)

        self.load_button = ttk.Button(control_frame, text="Load Video", command=self.load_video, bootstyle=PRIMARY)
        self.load_button.pack(side=LEFT, padx=5)

        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.pause_video, bootstyle=DANGER)
        self.pause_button.pack(side=LEFT, padx=5)

        self.play_button = ttk.Button(control_frame, text="Play", command=self.resume_video, bootstyle=SUCCESS)
        self.play_button.pack(side=LEFT, padx=5)

        self.restart_button = ttk.Button(control_frame, text="Restart", command=self.restart_video, bootstyle=INFO)
        self.restart_button.pack(side=LEFT, padx=5)

        # Slider de velocidad de reproducción
        speed_frame = ttk.Frame(self.root, padding=10)
        speed_frame.pack(fill=X, pady=5)

        speed_label = ttk.Label(speed_frame, text="Playback Speed", font=("Arial", 12))
        speed_label.pack(side=LEFT, padx=5)

        self.speed_slider = ttk.Scale(speed_frame, from_=0.5, to=2.0, variable=self.play_speed, orient=HORIZONTAL, length=200, bootstyle=SUCCESS)
        self.speed_slider.pack(side=LEFT, padx=10)

        # Canvas para el video (tamaño fijo)
        video_frame = ttk.Frame(self.root, padding=10)
        video_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.canvas_width = 640
        self.canvas_height = 480
        self.canvas = tk.Canvas(video_frame, bg="black", highlightthickness=2, highlightbackground="#d3d3d3",
                                 width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<MouseWheel>", self.canvas_zoom)

        # Panel de métricas
        self.metrics_frame = ttk.Frame(self.root, padding=10)
        self.metrics_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.metrics_label = ttk.Label(self.metrics_frame, text="Metrics Dashboard", font=("Arial", 14, "bold"))
        self.metrics_label.pack(pady=10)

        # Checkboxes para habilitar/deshabilitar métricas
        checkbox_frame = ttk.Frame(self.metrics_frame)
        checkbox_frame.pack(pady=5, fill=tk.X)

        for metric, var in self.selected_metrics.items():
            checkbox = ttk.Checkbutton(checkbox_frame, text=metric.replace("_", " ").title(), variable=var, bootstyle=SUCCESS)
            checkbox.pack(anchor=W)

        # Selector de modo de ángulo
        mode_frame = ttk.Frame(self.metrics_frame, padding=10)
        mode_frame.pack(pady=5, fill=tk.X)
        mode_label = ttk.Label(mode_frame, text="Angle Mode", font=("Arial", 12))
        mode_label.pack(anchor=W)

        relative_radio = ttk.Radiobutton(
            mode_frame, text="Relative", variable=self.mode, value="relative"
        )
        relative_radio.pack(anchor=W)

        fixed_radio = ttk.Radiobutton(
            mode_frame, text="Fixed", variable=self.mode, value="fixed"
        )
        fixed_radio.pack(anchor=W)

        # Selector de plano
        plane_frame = ttk.Frame(self.metrics_frame, padding=10)
        plane_frame.pack(pady=5, fill=tk.X)
        plane_label = ttk.Label(plane_frame, text="Reference Plane", font=("Arial", 12))
        plane_label.pack(anchor=W)

        horizontal_radio = ttk.Radiobutton(
            plane_frame, text="Horizontal", variable=self.plane, value="horizontal"
        )
        horizontal_radio.pack(anchor=W)

        vertical_radio = ttk.Radiobutton(
            plane_frame, text="Vertical", variable=self.plane, value="vertical"
        )
        vertical_radio.pack(anchor=W)

        # Crear gráficos
        self.create_graph()

        # Botón de salida
        exit_button = ttk.Button(self.root, text="Exit", command=self.root.quit, bootstyle=DANGER)
        exit_button.place(relx=0.95, rely=0.95, anchor=SE)

    def create_graph(self):
        """
        Crear un único gráfico para métricas seleccionadas.
        """
        graph_frame = ttk.Frame(self.root, padding=10)
        graph_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Agregar barra de herramientas para el gráfico
        toolbar = NavigationToolbar2Tk(self.graph_canvas, graph_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_graph(self, metrics):
        """
        Actualizar el gráfico en tiempo real con los datos calculados.
        """
        self.ax.clear()
        for metric, value in metrics.items():
            if metric in self.graph_data and self.selected_metrics[metric].get():
                self.graph_data[metric].append(value)
                self.ax.plot(self.graph_data[metric], label=metric.replace("_", " ").title())

        self.ax.set_xlim(0, max(100, max((len(data) for data in self.graph_data.values()), default=0)))
        self.ax.set_ylim(0, 180)
        self.ax.set_title("Angle Metrics Over Time")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Angle (°)")
        self.ax.legend()
        self.graph_canvas.draw()

    def canvas_click(self, event):
        """
        Maneja el clic del ratón sobre el canvas.
        """
        x, y = event.x, event.y
        canvas_coords = self.canvas_coords_to_video_coords(x, y)
        print(f"Canvas clicked at ({x}, {y}), Video coords: {canvas_coords}")

    def canvas_drag(self, event):
        """
        Maneja el arrastre del ratón sobre el canvas.
        """
        self.offset_x += event.x - self.canvas_width // 2
        self.offset_y += event.y - self.canvas_height // 2
        print(f"Dragging offset: ({self.offset_x}, {self.offset_y})")
        self.redraw_canvas()

    def canvas_zoom(self, event):
        """
        Maneja el zoom en el canvas con la rueda del ratón.
        """
        zoom_factor = 0.1
        if event.delta > 0:
            self.zoom_level += zoom_factor
        elif event.delta < 0 and self.zoom_level > zoom_factor:
            self.zoom_level -= zoom_factor
        print(f"Zoom level: {self.zoom_level}")
        self.redraw_canvas()

    def redraw_canvas(self):
        """
        Redibuja el contenido del canvas ajustado al zoom y desplazamiento.
        """
        # Aquí se debería ajustar la imagen en el canvas basado en zoom y desplazamiento
        pass

    def canvas_coords_to_video_coords(self, x, y):
        """
        Convierte coordenadas del canvas a las coordenadas del video considerando zoom y desplazamiento.
        """
        video_x = (x - self.offset_x) / self.zoom_level
        video_y = (y - self.offset_y) / self.zoom_level
        return int(video_x), int(video_y)

    def update_dashboard(self, metrics):
        """
        Actualiza el panel de métricas en tiempo real.
        :param metrics: Diccionario de métricas actuales.
        """
        print(f"Debug: Current metrics being processed: {metrics}")
        for key, value in metrics.items():
            if key in self.selected_metrics and self.selected_metrics[key].get():
                print(f"{key}: {value:.2f}° (Updating both video and graph)")
        self.update_graph(metrics)

    def load_video(self):
        """
        Abre un cuadro de diálogo para seleccionar un archivo de video.
        """
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if not self.video_path:
            messagebox.showwarning("No file selected", "Please select a video file!")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to open the video file!")
            return

        self.processed_frames = []  # Reiniciar los resultados procesados
        self.reset_graph_data()
        self.play_video()

    def reset_graph_data(self):
        """
        Reinicia los datos del gráfico.
        """
        for key in self.graph_data:
            self.graph_data[key] = []

    def play_video(self):
        """
        Reproduce el video fotograma por fotograma y sincroniza las métricas
        en el video y el dashboard.
        """
        if not self.cap:
            messagebox.showwarning("No video loaded", "Please load a video first!")
            return

        if self.is_paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        # Escalar el video mientras se mantiene la proporción
        video_height, video_width, _ = frame.shape
        scale = min(self.canvas_width / video_width, self.canvas_height / video_height)
        new_width = int(video_width * scale)
        new_height = int(video_height * scale)
        frame = cv2.resize(frame, (new_width, new_height))

        # Crear un fondo negro si el video no llena todo el canvas
        padded_frame = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        y_offset = (self.canvas_height - new_height) // 2
        x_offset = (self.canvas_width - new_width) // 2
        padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame

        # Procesar el fotograma
        results = self.processor.process_frame(padded_frame)

        if results.pose_landmarks:
            metrics = self.processor.calculate_metrics(
                results.pose_landmarks.landmark,
                mode=self.mode.get(),
                plane=self.plane.get()
            )

            self.processor.draw_landmarks(
                padded_frame, results,
                selected_metrics={key: var.get() for key, var in self.selected_metrics.items()},
                mode=self.mode.get(),
                plane=self.plane.get()
            )

            self.update_dashboard(metrics)

        # Mostrar el frame procesado
        padded_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(padded_frame))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

        if not self.is_paused:
            self.root.after(int(1000 / (30 * self.play_speed.get())), self.play_video)

    def pause_video(self):
        """
        Pausa la reproducción del video y detiene el registro de datos.
        """
        self.is_paused = True

    def resume_video(self):
        """
        Reanuda la reproducción del video después de una pausa.
        """
        if self.is_paused:
            self.is_paused = False
            self.play_video()

    def restart_video(self):
        """
        Reinicia la reproducción del video y los gráficos desde el inicio.
        """
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_paused = False
            self.reset_graph_data()
            self.play_video()
