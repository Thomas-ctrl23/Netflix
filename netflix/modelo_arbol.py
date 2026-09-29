# -*- coding: utf-8 -*-
"""
Módulo de Modelado y Clasificación con Árbol de Decisiones - Netflix
"""

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.arbol_model import NetflixDecisionTreePipeline

if __name__ == "__main__":
    print("Entrenando Árbol de Decisiones con datos ajustados...")
    pipeline = NetflixDecisionTreePipeline()
    metricas = pipeline.entrenar()
    print(f"Exactitud (Accuracy): {metricas['accuracy_test'] * 100:.2f}%")
    print(f"F1-Score:             {metricas['f1_test']:.4f}")
    print(f"Profundidad:          {metricas['profundidad_maxima']}")
    print(f"Hojas:                {metricas['n_hojas']}")
    pred = pipeline.predecir_2d(95)
    print(f"Predicción (95 min):  {pred['edad_recomendada']} años | {pred['clasificacion_sugerida']}")
