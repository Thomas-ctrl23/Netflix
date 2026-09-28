# -*- coding: utf-8 -*-
"""
Model: LimpiezaModel
Capa de Modelo para carga, sanitización y feature engineering del dataset de Netflix.
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

RATINGS_DESPLAZADOS = ["66 min", "74 min", "84 min"]

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
    df = df_crudo.copy()
    reporte = {
        "filas_originales": int(len(df)),
        "columnas_originales": list(df.columns)
    }

    # Imputación de nulos
    for col in ["director", "cast", "country", "date_added"]:
        if col in df.columns:
            df[col] = df[col].fillna("Desconocido")

    # Filtrar ratings desplazados y nulos
    if "rating" in df.columns:
        df = df[~df["rating"].isin(RATINGS_DESPLAZADOS)].copy()
        df = df.dropna(subset=["rating"])

        # Mapeo a target_age numérico
        df["target_age"] = df["rating"].map(AGE_MAP)
        df = df[df["target_age"].notnull()].copy()
        df["target_age"] = df["target_age"].astype(float)

    # Feature Engineering
    if "country" in df.columns:
        df["main_country"] = df["country"].astype(str).str.split(",").str[0].str.strip()
        df["main_country"] = df["main_country"].replace({"": "Desconocido", "nan": "Desconocido"})
    else:
        df["main_country"] = "Desconocido"

    if "listed_in" in df.columns:
        df["main_genre"] = df["listed_in"].astype(str).str.split(",").str[0].str.strip()
        df["main_genre"] = df["main_genre"].replace({"": "Desconocido", "nan": "Desconocido"})
    else:
        df["main_genre"] = "Desconocido"

    if "duration" in df.columns:
        df["duration_value"] = df["duration"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
        df["duration_unit"] = df["duration"].astype(str).str.extract(r"([A-Za-z]+)").fillna("min")
    else:
        df["duration_value"] = 0
        df["duration_unit"] = "min"

    if "type" in df.columns:
        df["is_movie"] = (df["type"] == "Movie").astype(int)
    else:
        df["is_movie"] = 1

    reporte["filas_limpias"] = int(len(df))
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
    return ruta_salida
