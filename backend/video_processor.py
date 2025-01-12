import cv2
import mediapipe as mp
import numpy as np
import math

class VideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Procesa un fotograma para detectar landmarks del cuerpo.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results

    def draw_landmarks(self, frame, results, selected_metrics, mode, plane, metrics):
        """
        Dibuja los landmarks del cuerpo y métricas seleccionadas en el fotograma.
        """
        if results.pose_landmarks:
            self.drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

            for metric, selected in selected_metrics.items():
                if selected and metric in metrics:
                    if mode == "relative":
                        point1, point2, point3 = self.get_points_for_metric(metric, results.pose_landmarks.landmark)
                        self.draw_angle_arc(frame, point1, point2, point3)
                    elif mode == "fixed":
                        point1, point2 = self.get_points_for_fixed_metric(metric, results.pose_landmarks.landmark, plane)
                        self.draw_fixed_angle_arc(frame, point1, point2, plane)
                    self.display_angle(
                        frame,
                        results.pose_landmarks.landmark[
                            self.get_landmark_for_metric(metric)
                        ],
                        metrics[metric]
                    )

    def get_landmark_for_metric(self, metric):
        """
        Retorna el landmark relevante para mostrar el ángulo calculado.
        """
        mapping = {
            "right_knee_angle": self.mp_pose.PoseLandmark.RIGHT_KNEE,
            "left_knee_angle": self.mp_pose.PoseLandmark.LEFT_KNEE,
            "right_shoulder_angle": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "left_shoulder_angle": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
        }
        return mapping.get(metric)

    def get_points_for_metric(self, metric, landmarks):
        """
        Retorna los tres puntos relevantes para calcular el ángulo de un métrico.
        """
        mapping = {
            "right_knee_angle": [
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ],
            "left_knee_angle": [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE
            ],
            "right_shoulder_angle": [
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ],
            "left_shoulder_angle": [
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP
            ]
        }
        points = mapping.get(metric)
        return [landmarks[points[0]], landmarks[points[1]], landmarks[points[2]]]

    def get_points_for_fixed_metric(self, metric, landmarks, plane):
        """
        Retorna los dos puntos relevantes para calcular el ángulo con un plano fijo.
        """
        mapping = {
            "right_knee_angle": [
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE
            ],
            "left_knee_angle": [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE
            ],
            "right_shoulder_angle": [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW
            ],
            "left_shoulder_angle": [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW
            ]
        }
        points = mapping.get(metric)
        return [landmarks[points[0]], landmarks[points[1]]]

    def draw_angle_arc(self, frame, point1, point2, point3):
        """
        Dibuja un arco entre tres puntos para visualizar el ángulo calculado.
        """
        # Convertir puntos a coordenadas de píxeles
        p1 = (int(point1.x * frame.shape[1]), int(point1.y * frame.shape[0]))
        p2 = (int(point2.x * frame.shape[1]), int(point2.y * frame.shape[0]))
        p3 = (int(point3.x * frame.shape[1]), int(point3.y * frame.shape[0]))

        # Calcular el radio aproximado y el centro del arco
        radius = int(np.linalg.norm(np.array(p1) - np.array(p2)) * 0.5)
        center = p2

        # Calcular los ángulos inicial y final del arco
        angle1 = math.degrees(math.atan2(p1[1] - center[1], p1[0] - center[0]))
        angle2 = math.degrees(math.atan2(p3[1] - center[1], p3[0] - center[0]))

        # Dibujar zona sombreada
        overlay = frame.copy()
        cv2.ellipse(overlay, center, (radius, radius), 0, angle1, angle2, (0, 255, 255), -1)
        alpha = 0.3  # Transparencia
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Dibujar el contorno del arco
        color = (0, 255, 255)  # Amarillo para el arco
        thickness = 2
        cv2.ellipse(frame, center, (radius, radius), 0, angle1, angle2, color, thickness)

    def draw_fixed_angle_arc(self, frame, point1, point2, plane):
        """
        Dibuja un arco para ángulos calculados con respecto a un plano fijo.
        """
        # Convertir puntos a coordenadas de píxeles
        p1 = (int(point1.x * frame.shape[1]), int(point1.y * frame.shape[0]))
        p2 = (int(point2.x * frame.shape[1]), int(point2.y * frame.shape[0]))

        # Determinar un punto en el plano para el arco
        if plane == "horizontal":
            p_plane = (p2[0] + 50, p2[1])  # Punto ficticio en la horizontal
        elif plane == "vertical":
            p_plane = (p2[0], p2[1] - 50)  # Punto ficticio en la vertical
        else:
            return

        # Calcular el radio aproximado
        radius = int(np.linalg.norm(np.array(p1) - np.array(p2)) * 0.5)
        center = p2

        # Calcular los ángulos inicial y final del arco
        angle1 = math.degrees(math.atan2(p1[1] - center[1], p1[0] - center[0]))
        angle2 = math.degrees(math.atan2(p_plane[1] - center[1], p_plane[0] - center[0]))

        # Dibujar zona sombreada
        overlay = frame.copy()
        cv2.ellipse(overlay, center, (radius, radius), 0, angle1, angle2, (255, 0, 255), -1)
        alpha = 0.3  # Transparencia
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Dibujar el contorno del arco
        color = (255, 0, 255)  # Magenta para el arco con plano fijo
        thickness = 2
        cv2.ellipse(frame, center, (radius, radius), 0, angle1, angle2, color, thickness)

    def calculate_angle_with_fixed_plane(self, point1, point2, plane="horizontal"):
        """
        Calcula el ángulo entre un segmento definido por dos puntos y un plano fijo (horizontal o vertical).

        :param point1: Coordenadas (x, y) del primer punto.
        :param point2: Coordenadas (x, y) del segundo punto.
        :param plane: "horizontal" o "vertical".
        :return: Ángulo en grados entre el segmento y el plano especificado.
        """
        x1, y1 = point1.x, point1.y
        x2, y2 = point2.x, point2.y

        # Vector del segmento
        dx = x2 - x1
        dy = y2 - y1

        try:
            if plane == "horizontal":
                # Calcular ángulo con la línea horizontal (y = constante)
                angle = math.atan2(abs(dy), abs(dx))  # atan2 considera el signo para determinar el cuadrante
            elif plane == "vertical":
                # Calcular ángulo con la línea vertical (x = constante)
                angle = math.atan2(abs(dx), abs(dy))
            else:
                raise ValueError("El plano debe ser 'horizontal' o 'vertical'.")

            return math.degrees(angle)
        except:
            return float('nan')

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
            metrics["right_knee_angle"] = self.calculate_joint_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE],
                mode, plane
            )
            metrics["left_knee_angle"] = self.calculate_joint_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE],
                mode, plane
            )
            metrics["right_shoulder_angle"] = self.calculate_joint_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                mode, plane
            )
            metrics["left_shoulder_angle"] = self.calculate_joint_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                mode, plane
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
            metrics["right_shoulder_angle"] = self.calculate_angle_with_fixed_plane(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                plane
            )
            metrics["left_shoulder_angle"] = self.calculate_angle_with_fixed_plane(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
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

        print(f"Debug: Calculated metrics: {metrics}")  # Depuración

        return metrics

    def calculate_joint_angle(self, point1, point2, point3, mode="relative", plane="horizontal"):
        """
        Calcula el ángulo entre tres puntos dados en un espacio 2D o 3D.
        """
        try:
            if plane == "horizontal":
                a = np.array([point1.x, point1.y])
                b = np.array([point2.x, point2.y])
                c = np.array([point3.x, point3.y])
            else:
                a = np.array([point1.x, point1.z])
                b = np.array([point2.x, point2.z])
                c = np.array([point3.x, point3.z])

            ab = a - b
            cb = c - b

            cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
            angle = np.degrees(np.arccos(cosine_angle))
            return angle
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return float('nan')  # Retorna NaN en caso de error

    def display_angle(self, frame, landmark, angle):
        """
        Muestra un ángulo en el fotograma cerca de un landmark específico.
        """
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])

        if not np.isnan(angle) and 0 <= angle <= 180:
            cv2.putText(
                frame, f"{angle:.1f}°", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                frame, "°", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
