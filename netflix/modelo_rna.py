# -*- coding: utf-8 -*-
"""
Módulo de Modelado y Clasificación con Redes Neuronales Artificiales (RNA / MLP) - Netflix
Entrena el modelo MLPRegressor (Duración -> Edad) y MLPClassifier usando Scikit-Learn
a partir de los datos preprocesados del catálogo de Netflix.
"""

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.rna_model import NetflixRNAPipeline

if __name__ == "__main__":
    print("=" * 70)
    print("REDES NEURONALES ARTIFICIALES (RNA / MLP) - NETFLIX")
    print("=" * 70)
    
    pipeline = NetflixRNAPipeline()
    metricas = pipeline.entrenar()

    print("\n[OK] Métricas de la Red Neuronal (RNA):")
    print(f"  - Registros de Entrenamiento: {metricas['n_train']}")
    print(f"  - Registros de Prueba:        {metricas['n_test']}")
    print(f"  - Coeficiente R² (Test):      {metricas['r2_test']}")
    print(f"  - Error Absoluto Medio (MAE): {metricas['mae_test']} años")
    print(f"  - Arquitectura de la Red:     {metricas['arquitectura']}")
    print(f"  - Función de Activación:      {metricas['activacion']}")
    print(f"  - Optimizador:                {metricas['optimizador']}")
    print(f"  - Épocas / Iteraciones:       {metricas['epocas_iter']}")
    print(f"  - Pérdida Final (Loss):       {metricas['perdida_final']}")

    pred_dur = pipeline.predecir_duracion(duracion_min=105)
    print(f"\n[OK] Predicción 2D RNA de Duración a Edad (105 min):")
    print(f"  - Edad recomendada:           {pred_dur['edad_recomendada']} años")
    print(f"  - Clasificación sugerida:     {pred_dur['clasificacion_sugerida']}")
    print(f"  - Detalle de la Red:          {pred_dur['formula_detalle']}")

    img_2d = pipeline.generar_grafico_duracion_edad("rna_duracion_edad_netflix.png")
    img_loss = pipeline.generar_grafico_perdida("rna_curva_perdida_netflix.png")
    img_mat = pipeline.generar_matriz_confusion("matriz_confusion_rna_netflix.png")
    print(f"\n[OK] Gráficos generados exitosamente:")
    print(f"  - Curva 2D Duración vs Edad:  {img_2d}")
    print(f"  - Curva de Pérdida (Loss):    {img_loss}")
    print(f"  - Matriz de Confusión:        {img_mat}")
    print("=" * 70)
