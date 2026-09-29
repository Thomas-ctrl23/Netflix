# -*- coding: utf-8 -*-
"""
Módulo de Modelado y Clasificación con Máquina de Vectores de Soporte (SVM) - Netflix
Entrena el modelo Support Vector Classifier (SVC) y Regresor (SVR) usando Scikit-Learn
a partir de los datos preprocesados del catálogo de Netflix.
"""

import os
import sys

# Asegurar importación de módulos internos
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.svm_model import NetflixSVMPipeline

if __name__ == "__main__":
    print("=" * 70)
    print("MÁQUINA DE VECTORES DE SOPORTE (SVM / SVR) - NETFLIX")
    print("=" * 70)
    
    pipeline = NetflixSVMPipeline(kernel="rbf", C=1.0)
    metricas = pipeline.entrenar()

    print("\n[OK] Métricas del Clasificador SVM:")
    print(f"  - Registros de Entrenamiento: {metricas['n_train']}")
    print(f"  - Registros de Prueba:        {metricas['n_test']}")
    print(f"  - Exactitud (Accuracy Test):  {metricas['accuracy_test'] * 100:.2f}%")
    print(f"  - F1-Score Ponderado:         {metricas['f1_test']:.4f}")
    print(f"  - Total Vectores de Soporte:  {metricas['total_vectores_soporte']}")
    print(f"  - Kernel:                     {metricas['kernel']}")
    print(f"  - Parámetro C:                {metricas['param_c']}")

    pred_dur = pipeline.predecir_duracion(duracion_min=105)
    print(f"\n[OK] Predicción 2D SVR de Duración a Edad (105 min):")
    print(f"  - Edad recomendada:           {pred_dur['edad_recomendada']} años")
    print(f"  - Clasificación sugerida:     {pred_dur['clasificacion_sugerida']}")
    print(f"  - Fórmula / Margen:           {pred_dur['formula_detalle']}")

    img_dur = pipeline.generar_grafico_duracion_edad("svm_duracion_edad_netflix.png")
    img_hip = pipeline.generar_grafico_hiperplano("svm_hiperplano_netflix.png")
    img_mat = pipeline.generar_matriz_confusion("matriz_confusion_svm_netflix.png")
    print(f"\n[OK] Gráficos generados exitosamente:")
    print(f"  - Curva 2D Duración vs Edad:  {img_dur}")
    print(f"  - Hiperplano 2D:              {img_hip}")
    print(f"  - Matriz de Confusión:        {img_mat}")
    print("=" * 70)
