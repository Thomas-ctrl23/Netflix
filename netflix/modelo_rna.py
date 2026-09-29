# -*- coding: utf-8 -*-
"""
Módulo de Modelado y Predicción con Redes Neuronales Artificiales (RNA / MLP) - Netflix
"""

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.rna_model import NetflixRNAPipeline

if __name__ == "__main__":
    print("Entrenando Red Neuronal Artificial (RNA) con datos ajustados...")
    pipeline = NetflixRNAPipeline()
    metricas = pipeline.entrenar()
    print(f"Coeficiente R² (Test): {metricas['r2_test']}")
    print(f"Error MAE (Test):     {metricas['mae_test']} años")
    print(f"Exactitud (Accuracy): {metricas['accuracy_clf'] * 100:.2f}%")
    print(f"Pérdida Final (Loss): {metricas['perdida_final']}")
    print(f"Arquitectura:         {metricas['arquitectura']}")
    pred = pipeline.predecir_duracion(95)
    print(f"Predicción (95 min):  {pred['edad_recomendada']} años | {pred['clasificacion_sugerida']}")
