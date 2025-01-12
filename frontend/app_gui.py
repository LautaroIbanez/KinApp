from PIL import Image, ImageTk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from backend.video_processor import VideoProcessor

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

        self.play_button = ttk.Button(control_frame, text="Play", command=self.play_video, bootstyle=SUCCESS)
        self.play_button.pack(side=LEFT, padx=5)

        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.pause_video, bootstyle=WARNING)
        self.pause_button.pack(side=LEFT, padx=5)

        self.restart_button = ttk.Button(control_frame, text="Restart", command=self.restart_video, bootstyle=INFO)
        self.restart_button.pack(side=LEFT, padx=5)

        # Canvas para el video
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black", highlightthickness=2, highlightbackground="#d3d3d3")
        self.canvas.pack(pady=10, side=tk.LEFT)

        # Panel de métricas
        self.metrics_frame = ttk.Frame(self.root, padding=10)
        self.metrics_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.metrics_label = ttk.Label(self.metrics_frame, text="Metrics Dashboard", font=("Arial", 14, "bold"))
        self.metrics_label.pack(pady=10)

        # Checkboxes para habilitar/deshabilitar métricas
        checkbox_frame = ttk.Frame(self.metrics_frame)
        checkbox_frame.pack(pady=5)

        for metric, var in self.selected_metrics.items():
            checkbox = ttk.Checkbutton(checkbox_frame, text=metric.replace("_", " ").title(), variable=var, bootstyle=SUCCESS)
            checkbox.pack(anchor=W)

        # Selector de modo de ángulo
        mode_frame = ttk.Frame(self.metrics_frame, padding=10)
        mode_frame.pack(pady=5)
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
        plane_frame.pack(pady=5)
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

    def update_dashboard(self, metrics):
        """
        Actualiza el panel de métricas en tiempo real.
        :param metrics: Diccionario de métricas actuales.
        """
        for key, value in metrics.items():
            if key in self.selected_metrics and self.selected_metrics[key].get():
                print(f"{key}: {value:.2f}°")

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
        self.play_video()

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

        # Procesar el fotograma
        results = self.processor.process_frame(frame)

        if results.pose_landmarks:
            metrics = self.processor.calculate_metrics(
                results.pose_landmarks.landmark,
                mode=self.mode.get(),
                plane=self.plane.get()
            )

            self.processor.draw_landmarks(
                frame, results,
                selected_metrics={key: var.get() for key, var in self.selected_metrics.items()},
                mode=self.mode.get(),
                plane=self.plane.get()
            )

            self.update_dashboard(metrics)

        # Mostrar el frame procesado
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

        if not self.is_paused:
            self.root.after(10, self.play_video)

    def pause_video(self):
        self.is_paused = True

    def restart_video(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_paused = False
            self.play_video()
