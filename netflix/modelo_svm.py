# -*- coding: utf-8 -*-
"""
Módulo de Modelado y Clasificación con Máquinas de Vectores de Soporte (SVM) - Netflix
"""

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.svm_model import NetflixSVMPipeline

if __name__ == "__main__":
    print("Entrenando SVM con datos ajustados y kernel RBF...")
    pipeline = NetflixSVMPipeline()
    metricas = pipeline.entrenar()
    print(f"Exactitud (Accuracy): {metricas['accuracy_test'] * 100:.2f}%")
    print(f"F1-Score:             {metricas['f1_test']:.4f}")
    print(f"Vectores de Soporte:  {metricas['total_vectores_soporte']}")
    pred = pipeline.predecir_duracion(95)
    print(f"Predicción (95 min):  {pred['edad_recomendada']} años | {pred['clasificacion_sugerida']}")
