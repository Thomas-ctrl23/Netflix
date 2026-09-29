# -*- coding: utf-8 -*-
"""
Módulo de Limpieza y Preprocesamiento de Datos - Netflix
Responsable de cargar el dataset crudo, imputar valores nulos,
sanitizar ratings, transformar la variable objetivo y generar
las variables de ingeniería (Feature Engineering) para alcanzar
parámetros de evaluación de alto rendimiento.
"""

import os
import sys
import json
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

# Categorías desplazadas, anómalas o sin calificación que distorsionan el aprendizaje
RATINGS_EXCLUIDOS = ["66 min", "74 min", "84 min", "Classic Movies, Documentaries", "NR", "UR"]

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
    Ejecuta el pipeline de ajuste, sanitización y Feature Engineering de datos.
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

    # 2. Filtrado de ratings desplazados y no clasificados (NR/UR sin calificación)
    if "rating" in df.columns:
        df = df[~df["rating"].isin(RATINGS_EXCLUIDOS)].copy()
        df = df.dropna(subset=["rating"])

        # 3. Mapeo a variable objetivo numérica continua 'target_age' (0 a 18 años)
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

    # 4. Unificación de duración a minutos reales y filtrado de valores atípicos
    if "duration" in df.columns:
        def extraer_minutos(row):
            d_str = str(row["duration"])
            num = float("".join(c for c in d_str if c.isdigit()) or 0)
            if "Season" in d_str:
                # 1 temporada promedio de serie = ~400 min de reproducción
                return num * 400.0
            return num

        df["duration_min"] = df.apply(extraer_minutos, axis=1)
        df = df[(df["duration_min"] >= 25) & (df["duration_min"] <= 4000)].copy()
        df["duration_value"] = df["duration_min"].astype(int)
        df["duration_unit"] = np.where(df["type"] == "Movie", "min", "Season")
    else:
        df["duration_min"] = 90.0
        df["duration_value"] = 90
        df["duration_unit"] = "min"

    # 5. Ingeniería de características (Feature Engineering)
    if "type" in df.columns:
        df["is_movie"] = (df["type"] == "Movie").astype(int)
    else:
        df["is_movie"] = 1

    if "listed_in" in df.columns:
        df["main_genre"] = df["listed_in"].astype(str).str.split(",").str[0].str.strip()
        df["main_genre"] = df["main_genre"].replace({"": "Desconocido", "nan": "Desconocido"})

        # Indicadores binarios de géneros de alto impacto en clasificación de audiencia
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

    # Análisis de señales semánticas en descripción
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

    # 6. Reporte de auditoría
    reporte["filas_limpias"] = int(len(df))
    reporte["filas_eliminadas"] = reporte["filas_originales"] - reporte["filas_limpias"]
    reporte["porcentaje_retencion"] = round((reporte["filas_limpias"] / max(reporte["filas_originales"], 1)) * 100, 2)
    reporte["distribucion_audiencia"] = df["audience_tier"].value_counts().to_dict() if "audience_tier" in df.columns else {}
    reporte["resumen_edad"] = {
        "min": float(df["target_age"].min()),
        "max": float(df["target_age"].max()),
        "promedio": round(float(df["target_age"].mean()), 2),
        "mediana": float(df["target_age"].median())
    }

    return df, reporte

def guardar_datos_limpios(df, ruta_salida=None, reporte=None):
    """Exporta el DataFrame limpio a formato CSV con codificación UTF-8."""
    if ruta_salida is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_salida = os.path.join(base_dir, "netflix_limpio.csv")

    df.to_csv(ruta_salida, index=False, encoding="utf-8")
    
    if reporte:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_reporte = os.path.join(base_dir, "reporte_limpieza.json")
        try:
            with open(ruta_reporte, "w", encoding="utf-8") as f:
                json.dump(reporte, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_raiz = os.path.join(parent_dir, "netflix_limpio.csv")
    try:
        df.to_csv(ruta_raiz, index=False, encoding="utf-8")
    except Exception:
        pass

    return ruta_salida

def ejecutar_pipeline_limpieza(ruta_entrada=None, ruta_salida=None):
    """Ejecuta el ciclo completo de carga, ajuste y exportación."""
    print("=" * 70)
    print("PIPELINE MODULAR DE AJUSTE Y LIMPIEZA DE DATOS - NETFLIX")
    print("=" * 70)

    df_crudo = cargar_dataset(ruta_entrada)
    print(f"[OK] Dataset cargado: {len(df_crudo)} filas, {df_crudo.shape[1]} columnas.")

    df_limpio, reporte = limpiar_datos(df_crudo)
    print(f"[OK] Limpieza y ajuste completados exitosamente.")
    print(f"     - Registros finales: {reporte['filas_limpias']} ({reporte['porcentaje_retencion']}% retenido)")
    print(f"     - Registros omitidos/desplazados: {reporte['filas_eliminadas']}")
    print(f"     - Rango de Edad: {reporte['resumen_edad']['min']} a {reporte['resumen_edad']['max']} anos (Media: {reporte['resumen_edad']['promedio']})")
    print(f"     - Distribución por Audiencia: {reporte.get('distribucion_audiencia', {})}")

    ruta_guardado = guardar_datos_limpios(df_limpio, ruta_salida, reporte)
    print(f"[OK] Archivo limpio guardado en: {ruta_guardado}")
    print("=" * 70)
    return df_limpio, reporte

if __name__ == "__main__":
    ejecutar_pipeline_limpieza()
