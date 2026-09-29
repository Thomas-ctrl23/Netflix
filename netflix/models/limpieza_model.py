# -*- coding: utf-8 -*-
"""
Model: LimpiezaModel
Capa de Modelo para carga, sanitización y feature engineering riguroso del dataset de Netflix.
Corrige inconsistencias de duración, elimina calificaciones anómalas y genera variables
predictoras enriquecidas para maximizar los parámetros de evaluación de los modelos de Machine Learning.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Diccionario canónico de mapeo de rating oficial a edad recomendada numérica
AGE_MAP = {
    "TV-Y": 0,
    "G": 0,
    "TV-G": 0,
    "TV-Y7": 7,
    "TV-Y7-FV": 7,
    "TV-PG": 10,
    "PG": 10,
    "PG-13": 13,
    "TV-14": 14,
    "TV-MA": 17,
    "R": 17,
    "NC-17": 18
}

# Categorías anómalas, desplazadas o sin calificación que distorsionan el aprendizaje
RATINGS_EXCLUIDOS = ["66 min", "74 min", "84 min", "Classic Movies, Documentaries", "NR", "UR"]

def obtener_ruta_dataset_predeterminada():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    if ruta_archivo is None:
        ruta_archivo = obtener_ruta_dataset_predeterminada()

    if ruta_archivo is None or not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_archivo}")

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
    Ejecuta el pipeline de ajuste, sanitización e ingeniería de datos.
    Garantiza que la duración esté unificada en minutos y que las variables
    categóricas y de texto aporten señal de alta correlación a los modelos.
    """
    df = df_crudo.copy()
    reporte = {
        "filas_originales": int(len(df)),
        "columnas_originales": list(df.columns)
    }

    # 1. Imputación inteligente de nulos
    for col in ["director", "cast", "country", "date_added"]:
        if col in df.columns:
            df[col] = df[col].fillna("Desconocido")

    # 2. Filtrar ratings desplazados y no clasificados (NR/UR sin calificación)
    if "rating" in df.columns:
        df = df[~df["rating"].isin(RATINGS_EXCLUIDOS)].copy()
        df = df.dropna(subset=["rating"])

        # Mapeo a target_age numérico oficial (0 a 18 años)
        df["target_age"] = df["rating"].map(AGE_MAP)
        df = df[df["target_age"].notnull()].copy()
        df["target_age"] = df["target_age"].astype(float)

        # Categorización canónica por audiencia (4 niveles estándar de la industria)
        def clasificar_audiencia(r):
            if r in ["TV-MA", "R", "NC-17"]:
                return "Adultos (+17)"
            elif r in ["TV-14", "PG-13"]:
                return "Adolescentes (+13)"
            elif r in ["TV-PG", "PG"]:
                return "Familiar (+10)"
            else:
                return "Infantil (TP)"

        df["audience_tier"] = df["rating"].apply(clasificar_audiencia)
        df["is_adult"] = df["rating"].isin(["TV-MA", "R", "NC-17"]).astype(int)

    # 3. Unificación y sanitización de la duración en minutos
    if "duration" in df.columns:
        def extraer_minutos(row):
            d_str = str(row["duration"])
            num = float("".join(c for c in d_str if c.isdigit()) or 0)
            if "Season" in d_str:
                # Cada temporada de TV equivale a ~400 minutos de contenido
                return num * 400.0
            return num

        df["duration_min"] = df.apply(extraer_minutos, axis=1)
        # Filtrar valores atípicos y extremos (duraciones menores a 25 min o series corruptas)
        df = df[(df["duration_min"] >= 25) & (df["duration_min"] <= 4000)].copy()
        df["duration_value"] = df["duration_min"].astype(int)
        df["duration_unit"] = np.where(df["type"] == "Movie", "min", "Season")
    else:
        df["duration_min"] = 90.0
        df["duration_value"] = 90
        df["duration_unit"] = "min"

    # 4. Feature Engineering: Variables de Contenido y Género
    if "type" in df.columns:
        df["is_movie"] = (df["type"] == "Movie").astype(int)
    else:
        df["is_movie"] = 1

    if "listed_in" in df.columns:
        df["main_genre"] = df["listed_in"].astype(str).str.split(",").str[0].str.strip()
        df["main_genre"] = df["main_genre"].replace({"": "Desconocido", "nan": "Desconocido"})
        
        # Indicadores binarios de géneros de alto impacto en la edad
        df["is_kids"] = df["listed_in"].str.contains("Children|Kids", case=False, na=False).astype(int)
        df["is_horror_crime"] = df["listed_in"].str.contains("Horror|Crime|Thriller", case=False, na=False).astype(int)
        df["is_drama"] = df["listed_in"].str.contains("Drama", case=False, na=False).astype(int)
        df["is_comedy"] = df["listed_in"].str.contains("Comed", case=False, na=False).astype(int)
        df["is_action"] = df["listed_in"].str.contains("Action", case=False, na=False).astype(int)
        df["is_doc"] = df["listed_in"].str.contains("Docu", case=False, na=False).astype(int)
        df["is_anime"] = df["listed_in"].str.contains("Anime", case=False, na=False).astype(int)
    else:
        df["main_genre"] = "Desconocido"
        for k in ["is_kids", "is_horror_crime", "is_drama", "is_comedy", "is_action", "is_doc", "is_anime"]:
            df[k] = 0

    if "country" in df.columns:
        df["main_country"] = df["country"].astype(str).str.split(",").str[0].str.strip()
        df["main_country"] = df["main_country"].replace({"": "Desconocido", "nan": "Desconocido"})
    else:
        df["main_country"] = "Desconocido"

    # 5. Feature Engineering: Palabras Clave de Madurez en Descripción
    if "description" in df.columns:
        desc = df["description"].fillna("").astype(str).str.lower()
        df["kw_violence"] = desc.str.contains("kill|murder|death|dead|blood|crime|investig|drug|gang|prison|soldier|war|cop|police|gun", regex=True).astype(int)
        df["kw_family"] = desc.str.contains("friend|school|magic|family|animal|puppy|dog|cat|cartoon|adventur|kid|child|toy|boy|girl", regex=True).astype(int)
        df["kw_romance"] = desc.str.contains("love|romance|marry|wedding|couple|relat|dating", regex=True).astype(int)
    else:
        df["kw_violence"] = 0
        df["kw_family"] = 0
        df["kw_romance"] = 0

    if "date_added" in df.columns:
        df["year_added"] = df["date_added"].astype(str).str.extract(r"(\d{4})").fillna(-1).astype(int)

    reporte["filas_limpias"] = int(len(df))
    reporte["filas_eliminadas"] = reporte["filas_originales"] - reporte["filas_limpias"]
    reporte["porcentaje_retencion"] = round((reporte["filas_limpias"] / max(reporte["filas_originales"], 1)) * 100, 2)
    reporte["resumen_edad"] = {
        "min": float(df["target_age"].min()),
        "max": float(df["target_age"].max()),
        "promedio": round(float(df["target_age"].mean()), 2),
        "mediana": float(df["target_age"].median())
    }
    reporte["distribucion_audiencia"] = df["audience_tier"].value_counts().to_dict() if "audience_tier" in df.columns else {}

    return df, reporte

def guardar_datos_limpios(df, ruta_salida=None, reporte=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ruta_salida is None:
        ruta_salida = os.path.join(base_dir, "netflix_limpio.csv")

    df.to_csv(ruta_salida, index=False, encoding="utf-8")
    if reporte:
        ruta_rep = os.path.join(base_dir, "reporte_limpieza.json")
        try:
            with open(ruta_rep, "w", encoding="utf-8") as f:
                json.dump(reporte, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    parent_dir = os.path.dirname(base_dir)
    ruta_raiz = os.path.join(parent_dir, "netflix_limpio.csv")
    try:
        df.to_csv(ruta_raiz, index=False, encoding="utf-8")
    except Exception:
        pass

    return ruta_salida
