# -*- coding: utf-8 -*-
"""
Pipeline de Regresión Lineal - Netflix
Predicción de la Edad Recomendada ('target_age') a partir de metadatos de contenido usando Scikit-Learn.
Refactorizado modularmente: la limpieza de datos está separada en 'netflix/limpieza_datos.py'.
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

# Asegurar codificación UTF-8 en salida de terminal en Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    print("=" * 70)
    print("PIPELINE DE MACHINE LEARNING - NETFLIX (MODULAR)")
    print("=" * 70)

    # 1. Carga del Dataset (Módulo Modular)
    print("\n1. CARGA DEL DATASET:")
    df_crudo = cargar_dataset()
    print(f"Dimensiones iniciales: {df_crudo.shape[0]} filas, {df_crudo.shape[1]} columnas.")

    # 2. Limpieza de Datos (Separada en netflix/limpieza_datos.py)
    print("\n2. LIMPIEZA Y TRANSFORMACIÓN A VARIABLE CONTINUA ('target_age'):")
    df_limpio, reporte = limpiar_datos(df_crudo)
    guardar_datos_limpios(df_limpio)
    print(f"Total de registros clasificados: {reporte['filas_limpias']} (Omitidos: {reporte['filas_eliminadas']})")
    print(f"Rango de edad recomendada: de {reporte['resumen_edad']['min']:.0f} a {reporte['resumen_edad']['max']:.0f} años.")
    print(f"Edad promedio en el catálogo: {reporte['resumen_edad']['promedio']:.2f} años (Mediana: {reporte['resumen_edad']['mediana']:.0f} años)")

    # 3. Modelado y Regresión (Separado en netflix/modelo_regresion.py)
    print("\n3. ENTRENAMIENTO DE MODELO DE REGRESIÓN LINEAL:")
    pipeline = NetflixRegressionPipeline()
    metricas = pipeline.entrenar(df_limpio)

    print(f"Entrenamiento (80%): {metricas['n_train']} registros")
    print(f"Prueba        (20%): {metricas['n_test']} registros")
    print(f"Intersección con el origen (b):          {metricas['intercepto']:.2f} años")
    print(f"Coeficiente de Determinación R² (Train): {metricas['r2_train']:.4f}")
    print(f"Coeficiente de Determinación R² (Test):  {metricas['r2_test']:.4f}")
    print(f"Error Absoluto Medio (MAE en Test):      {metricas['mae_test']:.2f} años de diferencia")
    print(f"Raíz del Error Cuadrático Medio (RMSE):   {metricas['rmse_test']:.2f} años")

    if "recta_2d" in metricas:
        print(f"\nEcuación de la recta 2D (Películas): {metricas['recta_2d']['ecuacion']}")

    # 4. Generación del Gráfico
    output_img = pipeline.generar_grafico_recta("regresion_lineal_netflix.png")
    print(f"\nGráfico con la línea recta guardado exitosamente como '{output_img}'.")
    print("=" * 70)
    print("Pipeline de Regresión Lineal finalizado exitosamente con código 0.")
    print("=" * 70)

if __name__ == "__main__":
    main()