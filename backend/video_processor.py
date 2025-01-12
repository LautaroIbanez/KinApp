import cv2
import mediapipe as mp
import math
import numpy as np

class VideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.drawing_utils = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Procesa un frame de video para detectar poses.
        :param frame: Imagen en formato BGR.
        :return: Resultados de detección de pose.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results

    def calculate_angle_with_fixed_plane(self, landmark1, landmark2, plane="horizontal"):
        """
        Calcula el ángulo de un segmento en relación con un plano fijo (horizontal o vertical).
        :param landmark1: Primer punto del segmento.
        :param landmark2: Segundo punto del segmento.
        :param plane: "horizontal" o "vertical".
        :return: Ángulo en grados respecto al plano seleccionado.
        """
        x1, y1 = landmark1.x, landmark1.y
        x2, y2 = landmark2.x, landmark2.y

        if plane == "horizontal":
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        elif plane == "vertical":
            angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
        else:
            raise ValueError("Invalid plane. Use 'horizontal' or 'vertical'.")

        angle = abs(angle)
        if angle > 90:
            angle = 180 - angle
        return angle

    def calculate_metrics(self, landmarks, mode="relative", plane="horizontal"):
        """
        Calcula las métricas relevantes a partir de los landmarks detectados.
        :param landmarks: Lista de puntos detectados.
        :param mode: Modo de cálculo de ángulos ("relative" o "fixed").
        :param plane: "horizontal" o "vertical" para cálculos fijos.
        :return: Diccionario de métricas calculadas.
        """
        metrics = {}

        if mode == "relative":
            # Ángulos relativos entre tres puntos
            metrics["right_knee_angle"] = self.calculate_angle_with_fixed_plane(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                plane
            )
            metrics["left_knee_angle"] = self.calculate_angle_with_fixed_plane(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
                plane
            )
        elif mode == "fixed":
            # Ángulos en relación con un plano fijo
            metrics["right_knee_angle"] = self.calculate_angle_with_fixed_plane(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                plane
            )
            metrics["left_knee_angle"] = self.calculate_angle_with_fixed_plane(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
                plane
            )

        # Simetría de las caderas y hombros
        metrics["hip_symmetry"] = abs(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y -
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y
        )
        metrics["shoulder_symmetry"] = abs(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y -
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        )

        return metrics

    def draw_landmarks(self, frame, results, selected_metrics, mode="relative", plane="horizontal"):
        """
        Dibuja puntos clave y métricas seleccionadas en el frame de video.
        """
        self.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
        )

        height, width, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Dibujar ángulos y ejes dinámicos
        if selected_metrics.get("right_knee_angle", False):
            self.draw_angle_with_axis(
                frame,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE],
                (0, 255, 0),  # Color
                "Right Knee Angle",  # Etiqueta
                mode,
                plane
            )

        if selected_metrics.get("left_knee_angle", False):
            self.draw_angle_with_axis(
                frame,
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE],
                (255, 0, 0),  # Color
                "Left Knee Angle",  # Etiqueta
                mode,
                plane
            )

    def draw_angle_with_axis(self, frame, landmark1, landmark2, landmark3, color, label, mode="relative", plane="horizontal"):
        """
        Dibuja el ángulo y los ejes en el frame de video.
        :param frame: Imagen en formato BGR.
        :param landmark1: Primer punto.
        :param landmark2: Segundo punto (vértice del ángulo).
        :param landmark3: Tercer punto.
        :param color: Color del texto y ejes.
        :param label: Etiqueta para el ángulo.
        :param mode: "relative" o "fixed" para el cálculo del ángulo.
        :param plane: "horizontal" o "vertical" para el plano fijo.
        """
        height, width, _ = frame.shape

        # Coordenadas en píxeles
        x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
        x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
        x3, y3 = int(landmark3.x * width), int(landmark3.y * height)

        # Calcular el ángulo
        if mode == "relative":
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        elif mode == "fixed":
            if plane == "horizontal":
                angle = math.degrees(math.atan2(y3 - y2, x3 - x2))
            elif plane == "vertical":
                angle = math.degrees(math.atan2(x3 - x2, y3 - y2))

        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle

        # Dibujar un arco dinámico para representar el ángulo
        radius = 50  # Radio del arco
        center = (x2, y2)
        if mode == "relative":
            start_angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            end_angle = math.degrees(math.atan2(y3 - y2, x3 - x2))
        elif mode == "fixed":
            if plane == "horizontal":
                start_angle = 0
                end_angle = angle
            elif plane == "vertical":
                start_angle = 90
                end_angle = 90 + angle

        overlay = frame.copy()
        cv2.ellipse(overlay, center, (radius, radius), 0, start_angle, end_angle, color, -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Dibujar texto del ángulo debajo del tag
        cv2.putText(
            frame,
            f"{label}: {angle:.1f}°",
            (x2 - 20, y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
