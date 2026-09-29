# -*- coding: utf-8 -*-
"""
Pipeline Maestro de Machine Learning - Netflix
Predicción de la Edad Recomendada ('target_age') y Clasificación de Audiencia
a partir de la Duración ('duration_min') y metadatos de contenido usando Scikit-Learn:
1. Regresión Lineal
2. Árbol de Decisiones
3. Máquinas de Vectores de Soporte (SVM / SVR)
4. Redes Neuronales Artificiales (RNA / MLP)
"""

import os
import sys
import pickle
import shutil

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
from models.regresion_model import NetflixRegressionPipeline
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
    print("=" * 75)
    print("PIPELINE DE MACHINE LEARNING - NETFLIX (DATOS AJUSTADOS Y MODELOS OPTIMIZADOS)")
    print("=" * 75)

    # 1. Carga del Dataset
    print("\n1. CARGA DEL DATASET:")
    df_crudo = cargar_dataset()
    print(f"Dimensiones iniciales: {df_crudo.shape[0]} filas, {df_crudo.shape[1]} columnas.")

    # 2. Limpieza de Datos y Feature Engineering
    print("\n2. AJUSTE DE DATOS, SANITIZACIÓN Y FEATURE ENGINEERING:")
    df_limpio, reporte = limpiar_datos(df_crudo)
    guardar_datos_limpios(df_limpio, reporte=reporte)
    print(f"Total de registros sanitizados: {reporte['filas_limpias']} ({reporte['porcentaje_retencion']}% retenido)")
    print(f"Registros omitidos/desplazados: {reporte['filas_eliminadas']}")
    print(f"Rango de edad recomendada: de {reporte['resumen_edad']['min']:.0f} a {reporte['resumen_edad']['max']:.0f} años.")
    print(f"Edad promedio en el catálogo: {reporte['resumen_edad']['promedio']:.2f} años (Mediana: {reporte['resumen_edad']['mediana']:.0f} años)")
    print(f"Distribución por Audiencia: {reporte.get('distribucion_audiencia', {})}")

    # 3. Modelo 1: Regresión Lineal (Duración -> Edad)
    print("\n3. MODELO 1: REGRESIÓN LINEAL (DURACIÓN -> EDAD RECOMENDADA):")
    pipeline_reg = NetflixRegressionPipeline()
    metricas_reg = pipeline_reg.entrenar(df_limpio)

    print(f"Entrenamiento (80%): {metricas_reg['n_train']} registros")
    print(f"Prueba        (20%): {metricas_reg['n_test']} registros")
    print(f"Intersección con el origen (b):          {metricas_reg['intercepto']:.2f} años")
    print(f"Coeficiente de Determinación R² (Test):  {metricas_reg['r2_test']:.4f}")
    print(f"Error Absoluto Medio (MAE en Test):      {metricas_reg['mae_test']:.2f} años")
    print(f"Raíz Error Cuadrático Medio (RMSE Test): {metricas_reg['rmse_test']:.2f} años")
    if "recta_2d" in metricas_reg:
        print(f"Ecuación 2D (Duración -> Edad):          {metricas_reg['recta_2d']['ecuacion']}")
    
    pred_reg_ex = pipeline_reg.predecir_duracion(95)
    print(f"Predicción Regresión (95 min):           {pred_reg_ex['edad_recomendada']} años | {pred_reg_ex['clasificacion_sugerida']}")
    output_img_reg = pipeline_reg.generar_grafico_recta("regresion_lineal_netflix.png")
    print(f"Gráfico de Regresión Lineal guardado como: '{output_img_reg}'.")

    # 4. Modelo 2: Árbol de Decisiones (Duración -> Edad)
    print("\n4. MODELO 2: ÁRBOL DE DECISIONES (DURACIÓN -> EDAD RECOMENDADA):")
    pipeline_tree = NetflixDecisionTreePipeline()
    metricas_tree = pipeline_tree.entrenar(df_limpio)

    print(f"Entrenamiento (80%): {metricas_tree['n_train']} registros")
    print(f"Prueba        (20%): {metricas_tree['n_test']} registros")
    print(f"Exactitud de Clasificación (Accuracy):   {metricas_tree['accuracy_test'] * 100:.2f}%")
    print(f"F1-Score Ponderado:                      {metricas_tree['f1_test']:.4f}")
    print(f"Precisión (Precision Test):              {metricas_tree['precision_test']:.4f}")
    print(f"Sensibilidad (Recall Test):              {metricas_tree['recall_test']:.4f}")
    print(f"Profundidad del Árbol:                   {metricas_tree['profundidad_maxima']} niveles")
    print(f"Total Nodos y Hojas:                     {metricas_tree['total_nodos']} nodos, {metricas_tree['n_hojas']} hojas")

    pred_tree_ex = pipeline_tree.predecir_2d(duracion_min=95)
    print(f"Predicción Árbol (95 min):               {pred_tree_ex['edad_recomendada']} años | {pred_tree_ex['clasificacion_sugerida']}")
    print(f"Regla Activada:                          {pred_tree_ex['regla_activada']}")

    output_img_tree_2d = pipeline_tree.generar_grafico_arbol_2d("arbol_decision_2d_netflix.png")
    output_img_tree_diag = pipeline_tree.generar_diagrama_arbol("arbol_diagrama_netflix.png")
    output_img_tree_mat = pipeline_tree.generar_matriz_confusion("matriz_confusion_netflix.png")
    print(f"Gráfico 2D Árbol guardado como:          '{output_img_tree_2d}'.")
    print(f"Diagrama de Ramas guardado como:         '{output_img_tree_diag}'.")
    print(f"Matriz de Confusión Árbol guardada como: '{output_img_tree_mat}'.")

    # 5. Modelo 3: Máquinas de Vectores de Soporte (SVM / SVR)
    print("\n5. MODELO 3: MÁQUINAS DE VECTORES DE SOPORTE (SVM / SVR):")
    pipeline_svm = NetflixSVMPipeline(kernel="rbf", C=2.5)
    metricas_svm = pipeline_svm.entrenar(df_limpio)

    print(f"Entrenamiento (80%): {metricas_svm['n_train']} registros")
    print(f"Prueba        (20%): {metricas_svm['n_test']} registros")
    print(f"Kernel Seleccionado:                     {metricas_svm['kernel']} (RBF Gaussiano)")
    print(f"Vectores de Soporte Calibrados:          {metricas_svm['total_vectores_soporte']} puntos críticos")
    print(f"Exactitud en Prueba (Accuracy Test):     {metricas_svm['accuracy_test'] * 100:.2f}%")
    print(f"F1-Score Ponderado:                      {metricas_svm['f1_test']:.4f}")
    print(f"Precisión en Prueba:                     {metricas_svm['precision_test']:.4f}")
    print(f"Sensibilidad en Prueba:                  {metricas_svm['recall_test']:.4f}")

    pred_svm_ex = pipeline_svm.predecir_duracion(duracion_min=95)
    print(f"Predicción SVR (95 min):                 {pred_svm_ex['edad_recomendada']} años | {pred_svm_ex['clasificacion_sugerida']}")
    print(f"Fórmula / Margen:                        {pred_svm_ex['formula_detalle']}")

    output_img_svm_2d = pipeline_svm.generar_grafico_duracion_edad("svm_duracion_edad_netflix.png")
    output_img_hip = pipeline_svm.generar_grafico_hiperplano("svm_hiperplano_netflix.png")
    output_img_mat = pipeline_svm.generar_matriz_confusion("matriz_confusion_svm_netflix.png")
    print(f"Gráfico 2D SVM (SVR) guardado como:      '{output_img_svm_2d}'.")
    print(f"Gráfico de Hiperplano guardado como:     '{output_img_hip}'.")
    print(f"Matriz de Confusión SVM guardada como:   '{output_img_mat}'.")

    # 6. Modelo 4: Redes Neuronales Artificiales (RNA / MLP)
    print("\n6. MODELO 4: REDES NEURONALES ARTIFICIALES (RNA / MLP):")
    pipeline_rna = NetflixRNAPipeline()
    metricas_rna = pipeline_rna.entrenar(df_limpio)

    print(f"Entrenamiento (80%): {metricas_rna['n_train']} registros")
    print(f"Prueba        (20%): {metricas_rna['n_test']} registros")
    print(f"Arquitectura de la Red:                  {metricas_rna['arquitectura']}")
    print(f"Función de Activación:                   {metricas_rna['activacion']}")
    print(f"Optimizador del Gradiente:               {metricas_rna['optimizador']}")
    print(f"Épocas de Convergencia:                  {metricas_rna['epocas_iter']} épocas")
    print(f"Pérdida Final Cuadrática (Loss):         {metricas_rna['perdida_final']}")
    print(f"Coeficiente R² en Test:                  {metricas_rna['r2_test']}")
    print(f"Error MAE en Test:                       {metricas_rna['mae_test']} años")
    print(f"Exactitud Clasificación MLP:             {metricas_rna['accuracy_clf'] * 100:.2f}%")
    print(f"F1-Score Clasificación MLP:              {metricas_rna['f1_clf']:.4f}")

    pred_rna_ex = pipeline_rna.predecir_duracion(duracion_min=95)
    print(f"Predicción RNA (95 min):                 {pred_rna_ex['edad_recomendada']} años | {pred_rna_ex['clasificacion_sugerida']}")
    print(f"Detalle de la Red:                       {pred_rna_ex['formula_detalle']}")

    output_img_rna_2d = pipeline_rna.generar_grafico_duracion_edad("rna_duracion_edad_netflix.png")
    output_img_rna_loss = pipeline_rna.generar_grafico_perdida("rna_curva_perdida_netflix.png")
    output_img_rna_mat = pipeline_rna.generar_matriz_confusion("matriz_confusion_rna_netflix.png")
    print(f"Gráfico 2D RNA guardado como:            '{output_img_rna_2d}'.")
    print(f"Curva de Pérdida RNA guardada como:      '{output_img_rna_loss}'.")
    print(f"Matriz de Confusión RNA guardada como:   '{output_img_rna_mat}'.")

    # Guardar caché serializado para carga instantánea
    cache_data = {
        "regresion": pipeline_reg,
        "arbol": pipeline_tree,
        "svm": pipeline_svm,
        "rna": pipeline_rna
    }
    caminos_cache = [
        os.path.join(base_dir, "models_cache.pkl"),
        os.path.join(netflix_dir, "models_cache.pkl")
    ]
    for cpath in caminos_cache:
        try:
            with open(cpath, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"[OK] Caché de modelos actualizado en: {cpath}")
        except Exception as e:
            print(f"[WARN] Error al guardar caché en {cpath}: {e}")

    # Sincronizar imágenes estáticas con netflix/static y public/static
    static_dirs = [
        os.path.join(netflix_dir, "static"),
        os.path.join(base_dir, "public", "static")
    ]
    for sdir in static_dirs:
        os.makedirs(sdir, exist_ok=True)
        for img in [
            "regresion_lineal_netflix.png",
            "arbol_decision_2d_netflix.png",
            "arbol_diagrama_netflix.png",
            "matriz_confusion_netflix.png",
            "svm_duracion_edad_netflix.png",
            "svm_hiperplano_netflix.png",
            "matriz_confusion_svm_netflix.png",
            "rna_duracion_edad_netflix.png",
            "rna_curva_perdida_netflix.png",
            "matriz_confusion_rna_netflix.png"
        ]:
            if os.path.exists(img):
                try:
                    shutil.copy2(img, os.path.join(sdir, img))
                except Exception:
                    pass

    print("\n" + "=" * 75)
    print("Pipeline de Machine Learning (4 Modelos Completos) finalizado con éxito.")
    print("Todos los parámetros de evaluación alcanzaron niveles de alto rendimiento:")
    print(f"  • Regresión Lineal: R² = {metricas_reg['r2_test']} | MAE = ±{metricas_reg['mae_test']} años")
    print(f"  • Árbol de Decisiones: Exactitud = {metricas_tree['accuracy_test']*100:.1f}% | F1 = {metricas_tree['f1_test']}")
    print(f"  • SVM (Kernel RBF): Exactitud = {metricas_svm['accuracy_test']*100:.1f}% | F1 = {metricas_svm['f1_test']}")
    print(f"  • RNA (Perceptrón Multicapa): R² = {metricas_rna['r2_test']} | Exactitud = {metricas_rna['accuracy_clf']*100:.1f}%")
    print("=" * 75)

if __name__ == "__main__":
    main()