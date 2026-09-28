# -*- coding: utf-8 -*-
"""
Módulo de Limpieza y Preprocesamiento de Datos - Netflix
Responsable de cargar el dataset crudo, imputar valores nulos,
sanitizar ratings, transformar la variable objetivo y generar
las variables de ingeniería (Feature Engineering).
"""

import os
import sys
import pandas as pd
import numpy as np

# Asegurar codificación UTF-8 en salida estándar
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Diccionario canónico de mapeo de rating oficial a edad recomendada numérica
AGE_MAP = {
    "TV-Y": 0,
    "G": 0,
    "TV-Y7": 7,
    "TV-Y7-FV": 7,
    "TV-PG": 8,
    "PG": 8,
    "PG-13": 13,
    "TV-14": 14,
    "TV-MA": 17,
    "R": 17,
    "NC-17": 18,
    "NR": 17,
    "UR": 17
}

# Categorías desplazadas o anómalas en la columna 'rating'
RATINGS_DESPLAZADOS = ["66 min", "74 min", "84 min"]

def obtener_ruta_dataset_predeterminada():
    """Busca el archivo CSV original en las rutas habituales del proyecto."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    candidatas = [
        os.path.join(base_dir, "Datos_CSV(NETFLIX).csv"),
        os.path.join(parent_dir, "Datos_CSV(NETFLIX).csv"),
        os.path.join(base_dir, "netflix_titles.csv"),
        os.path.join(parent_dir, "netflix_titles.csv"),
        os.path.join(parent_dir, "netflix_titles (1).csv"),
        "Datos_CSV(NETFLIX).csv",
        "netflix/Datos_CSV(NETFLIX).csv"
    ]
    for ruta in candidatas:
        if os.path.exists(ruta):
            return ruta
    return None

def cargar_dataset(ruta_archivo=None):
    """
    Carga el archivo CSV crudo con manejo inteligente de codificación y delimitador.
    Retorna: pd.DataFrame
    """
    if ruta_archivo is None:
        ruta_archivo = obtener_ruta_dataset_predeterminada()

    if ruta_archivo is None or not os.path.exists(ruta_archivo):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos de Netflix. Ruta especificada o buscada: {ruta_archivo}"
        )

    # Intentar con separador ';' y latin1 (formato original del archivo), luego fallback a coma y utf-8
    try:
        df = pd.read_csv(ruta_archivo, sep=";", encoding="latin1")
        if df.shape[1] <= 1:
            df = pd.read_csv(ruta_archivo, sep=",", encoding="latin1")
    except Exception:
        try:
            df = pd.read_csv(ruta_archivo, sep=",", encoding="utf-8", errors="replace")
        except Exception:
            df = pd.read_csv(ruta_archivo, sep=";", encoding="utf-8", errors="replace")

    return df

def limpiar_datos(df_crudo):
    """
    Ejecuta el pipeline de limpieza y transformación de datos.
    Retorna: (df_limpio, reporte_limpieza)
    """
    df = df_crudo.copy()
    reporte = {
        "filas_originales": int(len(df)),
        "columnas_originales": list(df.columns),
        "nulos_por_columna_antes": df.isnull().sum().to_dict()
    }

    # 1. Imputación de columnas con valores nulos
    imputaciones = {
        "director": "Desconocido",
        "cast": "Desconocido",
        "country": "Desconocido",
        "date_added": "Desconocido"
    }
    for col, val in imputaciones.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # 2. Filtrado de ratings desplazados / anomalías de extracción
    if "rating" in df.columns:
        df = df[~df["rating"].isin(RATINGS_DESPLAZADOS)].copy()

    # 3. Mapeo a variable objetivo numérica continua 'target_age'
    if "rating" in df.columns:
        df["target_age"] = df["rating"].map(AGE_MAP)
        # Descartar registros que no tengan un rating mapeable
        filas_con_target = df["target_age"].notnull()
        df = df[filas_con_target].copy()
        df["target_age"] = df["target_age"].astype(float)

    # 4. Ingeniería de características (Feature Engineering)
    # País principal (primer país listado)
    if "country" in df.columns:
        df["main_country"] = df["country"].astype(str).str.split(",").str[0].str.strip()
        df["main_country"] = df["main_country"].replace({"": "Desconocido", "nan": "Desconocido"})
    else:
        df["main_country"] = "Desconocido"

    # Género principal (primer género de 'listed_in')
    if "listed_in" in df.columns:
        df["main_genre"] = df["listed_in"].astype(str).str.split(",").str[0].str.strip()
        df["main_genre"] = df["main_genre"].replace({"": "Desconocido", "nan": "Desconocido"})
    else:
        df["main_genre"] = "Desconocido"

    # Duración numérica y unidad
    if "duration" in df.columns:
        df["duration_value"] = df["duration"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
        df["duration_unit"] = df["duration"].astype(str).str.extract(r"([A-Za-z]+)").fillna("min")
    else:
        df["duration_value"] = 0
        df["duration_unit"] = "min"

    # Indicador binario de película vs serie
    if "type" in df.columns:
        df["is_movie"] = (df["type"] == "Movie").astype(int)
    else:
        df["is_movie"] = 1

    # Limpieza y extracción del año de agregado si está presente
    if "date_added" in df.columns:
        df["year_added"] = df["date_added"].astype(str).str.extract(r"(\d{4})").fillna(-1).astype(int)

    # 5. Generación del reporte de auditoría
    reporte["filas_limpias"] = int(len(df))
    reporte["filas_eliminadas"] = reporte["filas_originales"] - reporte["filas_limpias"]
    reporte["porcentaje_retencion"] = round((reporte["filas_limpias"] / max(reporte["filas_originales"], 1)) * 100, 2)
    reporte["distribucion_ratings"] = df["rating"].value_counts().to_dict() if "rating" in df.columns else {}
    reporte["resumen_edad"] = {
        "min": float(df["target_age"].min()) if "target_age" in df.columns else 0,
        "max": float(df["target_age"].max()) if "target_age" in df.columns else 0,
        "promedio": round(float(df["target_age"].mean()), 2) if "target_age" in df.columns else 0,
        "mediana": float(df["target_age"].median()) if "target_age" in df.columns else 0
    }
    reporte["conteo_tipos"] = df["type"].value_counts().to_dict() if "type" in df.columns else {}
    reporte["top_paises"] = df["main_country"].value_counts().head(10).to_dict() if "main_country" in df.columns else {}
    reporte["top_generos"] = df["main_genre"].value_counts().head(10).to_dict() if "main_genre" in df.columns else {}

    return df, reporte

def guardar_datos_limpios(df, ruta_salida=None, reporte=None):
    """
    Exporta el DataFrame limpio a formato CSV con codificación UTF-8.
    """
    import json
    if ruta_salida is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_salida = os.path.join(base_dir, "netflix_limpio.csv")

    df.to_csv(ruta_salida, index=False, encoding="utf-8")
    
    # Si se provee reporte, guardarlo en JSON
    if reporte:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_reporte = os.path.join(base_dir, "reporte_limpieza.json")
        try:
            with open(ruta_reporte, "w", encoding="utf-8") as f:
                json.dump(reporte, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # También guardar una copia en el directorio raíz para acceso directo
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_raiz = os.path.join(parent_dir, "netflix_limpio.csv")
    try:
        df.to_csv(ruta_raiz, index=False, encoding="utf-8")
    except Exception:
        pass

    return ruta_salida

def ejecutar_pipeline_limpieza(ruta_entrada=None, ruta_salida=None):
    """
    Ejecuta el ciclo completo de carga, limpieza y exportación,
    imprimiendo estadísticas en consola.
    """
    print("=" * 70)
    print("PIPELINE MODULAR DE LIMPIEZA DE DATOS - NETFLIX")
    print("=" * 70)

    df_crudo = cargar_dataset(ruta_entrada)
    print(f"[OK] Dataset cargado: {len(df_crudo)} filas, {df_crudo.shape[1]} columnas.")

    df_limpio, reporte = limpiar_datos(df_crudo)
    print(f"[OK] Limpieza completada exitosamente.")
    print(f"     - Registros finales: {reporte['filas_limpias']} ({reporte['porcentaje_retencion']}% retenido)")
    print(f"     - Registros omitidos/desplazados: {reporte['filas_eliminadas']}")
    print(f"     - Rango de Edad: {reporte['resumen_edad']['min']} a {reporte['resumen_edad']['max']} a単os (Media: {reporte['resumen_edad']['promedio']})")
    print(f"     - Distribución Contenido: {reporte['conteo_tipos']}")

    ruta_guardado = guardar_datos_limpios(df_limpio, ruta_salida, reporte)
    print(f"[OK] Archivo limpio guardado en: {ruta_guardado}")
    print("=" * 70)
    return df_limpio, reporte

if __name__ == "__main__":
    ejecutar_pipeline_limpieza()
