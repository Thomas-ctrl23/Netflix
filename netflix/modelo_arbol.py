# -*- coding: utf-8 -*-
"""
Módulo de Modelado y Clasificación con Árbol de Decisiones - Netflix
Entrena el modelo DecisionTreeClassifier y DecisionTreeRegressor usando Scikit-Learn
a partir de los datos preprocesados del catálogo de Netflix.
"""

import os
import sys

# Asegurar importación de módulos internos
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from models.arbol_model import NetflixDecisionTreePipeline

if __name__ == "__main__":
    print("=" * 70)
    print("ÁRBOL DE DECISIONES - NETFLIX (DURACIÓN -> EDAD RECOMENDADA)")
    print("=" * 70)
    
    pipeline = NetflixDecisionTreePipeline()
    metricas = pipeline.entrenar()

    print("\n[OK] Métricas del Árbol de Decisiones:")
    print(f"  - Registros de Entrenamiento: {metricas['n_train']}")
    print(f"  - Registros de Prueba:        {metricas['n_test']}")
    print(f"  - Exactitud (Accuracy Test):  {metricas['accuracy_test'] * 100:.2f}%")
    print(f"  - F1-Score Ponderado:         {metricas['f1_test']:.4f}")
    print(f"  - Profundidad Máxima:         {metricas['profundidad_maxima']}")
    print(f"  - Total de Nodos:             {metricas['total_nodos']}")
    print(f"  - Hojas Finales:              {metricas['n_hojas']}")

    pred_2d = pipeline.predecir_2d(duracion_min=95)
    print(f"\n[OK] Predicción de Duración a Edad (95 min):")
    print(f"  - Edad recomendada:           {pred_2d['edad_recomendada']} años")
    print(f"  - Clasificación sugerida:     {pred_2d['clasificacion_sugerida']}")
    print(f"  - Regla activada:             {pred_2d['regla_activada']}")

    img_2d = pipeline.generar_grafico_arbol_2d("arbol_decision_2d_netflix.png")
    img_diag = pipeline.generar_diagrama_arbol("arbol_diagrama_netflix.png")
    img_mat = pipeline.generar_matriz_confusion("matriz_confusion_netflix.png")
    print(f"\n[OK] Gráficos generados exitosamente:")
    print(f"  - Curva 2D Duración vs Edad:  {img_2d}")
    print(f"  - Diagrama de Ramas:          {img_diag}")
    print(f"  - Matriz de Confusión:        {img_mat}")
    print("=" * 70)
