# -*- coding: utf-8 -*-
"""
Pipeline Maestro de Machine Learning - Netflix
Predicción de la Edad Recomendada ('target_age') a partir de la Duración ('duration_value')
y metadatos de contenido usando Scikit-Learn:
1. Regresión Lineal
2. Árbol de Decisiones
3. Máquinas de Vectores de Soporte (SVM / SVR)
"""

import os
import sys

# Asegurar importación de los módulos de la carpeta netflix
base_dir = os.path.dirname(os.path.abspath(__file__))
netflix_dir = os.path.join(base_dir, "netflix")
if netflix_dir not in sys.path:
    sys.path.insert(0, netflix_dir)

# Importar módulos modulares
from limpieza_datos import (
    cargar_dataset,
    limpiar_datos,
    guardar_datos_limpios
)
from modelo_regresion import NetflixRegressionPipeline
from models.arbol_model import NetflixDecisionTreePipeline
from models.svm_model import NetflixSVMPipeline
from models.rna_model import NetflixRNAPipeline

# Asegurar codificación UTF-8 en salida de terminal en Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    print("=" * 70)
    print("PIPELINE DE MACHINE LEARNING - NETFLIX (3 MODELOS: DURACIÓN -> EDAD)")
    print("=" * 70)

    # 1. Carga del Dataset
    print("\n1. CARGA DEL DATASET:")
    df_crudo = cargar_dataset()
    print(f"Dimensiones iniciales: {df_crudo.shape[0]} filas, {df_crudo.shape[1]} columnas.")

    # 2. Limpieza de Datos
    print("\n2. LIMPIEZA Y TRANSFORMACIÓN A VARIABLE CONTINUA ('target_age'):")
    df_limpio, reporte = limpiar_datos(df_crudo)
    guardar_datos_limpios(df_limpio)
    print(f"Total de registros clasificados: {reporte['filas_limpias']} (Omitidos: {reporte['filas_eliminadas']})")
    print(f"Rango de edad recomendada: de {reporte['resumen_edad']['min']:.0f} a {reporte['resumen_edad']['max']:.0f} años.")
    print(f"Edad promedio en el catálogo: {reporte['resumen_edad']['promedio']:.2f} años (Mediana: {reporte['resumen_edad']['mediana']:.0f} años)")

    # 3. Modelo 1: Regresión Lineal (Duración -> Edad)
    print("\n3. MODELO 1: REGRESIÓN LINEAL (DURACIÓN -> EDAD RECOMENDADA):")
    pipeline_reg = NetflixRegressionPipeline()
    metricas_reg = pipeline_reg.entrenar(df_limpio)

    print(f"Entrenamiento (80%): {metricas_reg['n_train']} registros")
    print(f"Prueba        (20%): {metricas_reg['n_test']} registros")
    print(f"Intersección con el origen (b):          {metricas_reg['intercepto']:.2f} años")
    print(f"Coeficiente de Determinación R² (Test):  {metricas_reg['r2_test']:.4f}")
    print(f"Error Absoluto Medio (MAE en Test):      {metricas_reg['mae_test']:.2f} años")
    if "recta_2d" in metricas_reg:
        print(f"Ecuación 2D (Duración -> Edad):          {metricas_reg['recta_2d']['ecuacion']}")
    
    pred_reg_ex = pipeline_reg.predecir_duracion(105) if hasattr(pipeline_reg, "predecir_duracion") else {"edad": 14.14}
    output_img_reg = pipeline_reg.generar_grafico_recta("regresion_lineal_netflix.png")
    print(f"Gráfico de Regresión Lineal guardado como: '{output_img_reg}'.")

    # 4. Modelo 2: Árbol de Decisiones (Duración -> Edad)
    print("\n4. MODELO 2: ÁRBOL DE DECISIONES (DURACIÓN -> EDAD RECOMENDADA):")
    pipeline_tree = NetflixDecisionTreePipeline()
    metricas_tree = pipeline_tree.entrenar(df_limpio)

    print(f"Entrenamiento (80%): {metricas_tree['n_train']} registros")
    print(f"Prueba        (20%): {metricas_tree['n_test']} registros")
    print(f"Exactitud de Clasificación (Accuracy):   {metricas_tree['accuracy_test'] * 100:.2f}%")
    print(f"F1-Score Multiclase:                     {metricas_tree['f1_test']:.4f}")
    print(f"Profundidad Máxima del Árbol:            {metricas_tree['profundidad_maxima']}")
    print(f"Total Nodos y Hojas:                     {metricas_tree['total_nodos']} nodos, {metricas_tree['n_hojas']} hojas")

    pred_tree_ex = pipeline_tree.predecir_2d(duracion_min=105)
    print(f"Predicción Árbol (105 min):              {pred_tree_ex['edad_recomendada']} años | {pred_tree_ex['clasificacion_sugerida']}")
    print(f"Regla Activada:                          {pred_tree_ex['regla_activada']}")

    output_img_tree_2d = pipeline_tree.generar_grafico_arbol_2d("arbol_decision_2d_netflix.png")
    output_img_tree_diag = pipeline_tree.generar_diagrama_arbol("arbol_diagrama_netflix.png")
    print(f"Gráfico 2D Árbol guardado como:          '{output_img_tree_2d}'.")
    print(f"Diagrama de Ramas guardado como:         '{output_img_tree_diag}'.")

    # 5. Modelo 3: Máquinas de Vectores de Soporte (SVM / SVR) (Duración -> Edad)
    print("\n5. MODELO 3: MÁQUINAS DE VECTORES DE SOPORTE (SVM / SVR):")
    pipeline_svm = NetflixSVMPipeline(kernel="rbf", C=1.0)
    metricas_svm = pipeline_svm.entrenar(df_limpio)

    print(f"Entrenamiento (80%): {metricas_svm['n_train']} registros")
    print(f"Prueba        (20%): {metricas_svm['n_test']} registros")
    print(f"Kernel Seleccionado:                     {metricas_svm['kernel']} (RBF Gaussiano)")
    print(f"Total Vectores de Soporte:               {metricas_svm['total_vectores_soporte']} puntos críticos")
    print(f"Exactitud en Prueba (Accuracy Test):     {metricas_svm['accuracy_test'] * 100:.2f}%")
    print(f"F1-Score Multiclase Ponderado:           {metricas_svm['f1_test']:.4f}")

    pred_svm_ex = pipeline_svm.predecir_duracion(duracion_min=105)
    print(f"Predicción SVR (105 min):                {pred_svm_ex['edad_recomendada']} años | {pred_svm_ex['clasificacion_sugerida']}")
    print(f"Fórmula / Margen:                        {pred_svm_ex['formula_detalle']}")

    output_img_svm_2d = pipeline_svm.generar_grafico_duracion_edad("svm_duracion_edad_netflix.png")
    output_img_hip = pipeline_svm.generar_grafico_hiperplano("svm_hiperplano_netflix.png")
    output_img_mat = pipeline_svm.generar_matriz_confusion("matriz_confusion_svm_netflix.png")
    print(f"Gráfico 2D SVM (SVR) guardado como:      '{output_img_svm_2d}'.")
    print(f"Gráfico de Hiperplano guardado como:     '{output_img_hip}'.")
    print(f"Matriz de Confusión guardada como:       '{output_img_mat}'.")

    # 6. Modelo 4: Redes Neuronales Artificiales (RNA / MLP) (Duración -> Edad)
    print("\n6. MODELO 4: REDES NEURONALES ARTIFICIALES (RNA / MLP):")
    pipeline_rna = NetflixRNAPipeline()
    metricas_rna = pipeline_rna.entrenar(df_limpio)

    print(f"Entrenamiento (80%): {metricas_rna['n_train']} registros")
    print(f"Prueba        (20%): {metricas_rna['n_test']} registros")
    print(f"Arquitectura de la Red:                  {metricas_rna['arquitectura']}")
    print(f"Función de Activación:                   {metricas_rna['activacion']}")
    print(f"Optimizador del Gradiente:               {metricas_rna['optimizador']}")
    print(f"Épocas de Convergencia:                  {metricas_rna['epocas_iter']}")
    print(f"Pérdida Final Cuadrática (Loss):         {metricas_rna['perdida_final']}")
    print(f"Coeficiente R² en Test:                  {metricas_rna['r2_test']}")
    print(f"Error MAE en Test:                       {metricas_rna['mae_test']} años")

    pred_rna_ex = pipeline_rna.predecir_duracion(duracion_min=105)
    print(f"Predicción RNA (105 min):                {pred_rna_ex['edad_recomendada']} años | {pred_rna_ex['clasificacion_sugerida']}")
    print(f"Detalle de la Red:                       {pred_rna_ex['formula_detalle']}")

    output_img_rna_2d = pipeline_rna.generar_grafico_duracion_edad("rna_duracion_edad_netflix.png")
    output_img_rna_loss = pipeline_rna.generar_grafico_perdida("rna_curva_perdida_netflix.png")
    output_img_rna_mat = pipeline_rna.generar_matriz_confusion("matriz_confusion_rna_netflix.png")
    print(f"Gráfico 2D RNA guardado como:            '{output_img_rna_2d}'.")
    print(f"Curva de Pérdida RNA guardada como:      '{output_img_rna_loss}'.")
    print(f"Matriz de Confusión RNA guardada como:   '{output_img_rna_mat}'.")

    print("\n" + "=" * 70)
    print("Pipeline de Machine Learning (4 Modelos Completos) finalizado con éxito.")
    print("Para interactuar en la Web abre: http://127.0.0.1:5000")
    print("=" * 70)

if __name__ == "__main__":
    main()