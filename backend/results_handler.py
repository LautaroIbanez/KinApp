import os
import json
import csv

class ResultsHandler:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  # Crear la carpeta si no existe

    def save_to_json(self, data, filename="results.json"):
        """
        Guarda los datos en un archivo JSON.
        :param data: Datos a guardar (diccionario o lista).
        :param filename: Nombre del archivo de salida.
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Results saved to {filepath}")

    def save_to_csv(self, data, filename="results.csv"):
        """
        Guarda los datos en un archivo CSV.
        :param data: Lista de diccionarios (cada diccionario es una fila).
        :param filename: Nombre del archivo de salida.
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"Results saved to {filepath}")
