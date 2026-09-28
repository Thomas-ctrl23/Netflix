# -*- coding: utf-8 -*-
"""
Módulo de Modelado y Regresión Lineal - Netflix
Entrena el modelo de Regresión Lineal Múltiple usando Scikit-Learn
a partir de los datos preprocesados por el módulo de limpieza.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Evita inicialización de GUI interactiva
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Asegurar importación del módulo de limpieza local
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    from limpieza_datos import cargar_dataset, limpiar_datos
except ImportError:
    from netflix.limpieza_datos import cargar_dataset, limpiar_datos

CATEGORICAL_FEATURES = ["type", "main_country", "duration_unit", "main_genre"]
NUMERIC_FEATURES = ["release_year", "duration_value"]

class NetflixRegressionPipeline:
    def __init__(self):
        self.preprocessor = None
        self.modelo_multiple = None
        self.modelo_simple_duracion = None
        self.metricas = {}
        self.is_fitted = False
        self.df_clean = None

    def entrenar(self, df=None):
        """Entrena los modelos de regresión y almacena métricas."""
        if df is None:
            df_crudo = cargar_dataset()
            df, _ = limpiar_datos(df_crudo)

        self.df_clean = df.copy()

        # Preparar variables
        X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        y = df["target_age"]

        # Partición 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # Preprocesador
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
                ("num", StandardScaler(), NUMERIC_FEATURES)
            ]
        )

        X_train_encoded = self.preprocessor.fit_transform(X_train)
        X_test_encoded = self.preprocessor.transform(X_test)

        # Modelo Múltiple
        self.modelo_multiple = LinearRegression()
        self.modelo_multiple.fit(X_train_encoded, y_train)

        y_train_pred = self.modelo_multiple.predict(X_train_encoded)
        y_test_pred = self.modelo_multiple.predict(X_test_encoded)

        self.metricas = {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "intercepto": round(float(self.modelo_multiple.intercept_), 2),
            "r2_train": round(float(r2_score(y_train, y_train_pred)), 4),
            "r2_test": round(float(r2_score(y_test, y_test_pred)), 4),
            "mae_test": round(float(mean_absolute_error(y_test, y_test_pred)), 2),
            "rmse_test": round(float(root_mean_squared_error(y_test, y_test_pred)), 2)
        }

        # Modelo Simple 2D (Películas: Duración vs Edad)
        movies_mask = df["type"] == "Movie"
        X_movie_dur = df.loc[movies_mask, ["duration_value"]].values
        y_movie_age = df.loc[movies_mask, "target_age"].values

        if len(X_movie_dur) > 0:
            self.modelo_simple_duracion = LinearRegression()
            self.modelo_simple_duracion.fit(X_movie_dur, y_movie_age)
            m_pen = float(self.modelo_simple_duracion.coef_[0])
            b_orig = float(self.modelo_simple_duracion.intercept_)
            self.metricas["recta_2d"] = {
                "pendiente": round(m_pen, 4),
                "intercepto": round(b_orig, 2),
                "ecuacion": f"Edad = ({m_pen:.4f} * Duración) + {b_orig:.2f}"
            }

        self.is_fitted = True
        return self.metricas

    def predecir(self, tipo="Movie", pais="United States", genero="Dramas", anio=2021, duracion=95):
        """
        Predice la edad recomendada para un título dado sus metadatos.
        """
        if not self.is_fitted:
            self.entrenar()

        unidad = "min" if tipo == "Movie" else "Season"
        df_input = pd.DataFrame([{
            "type": tipo,
            "main_country": pais,
            "duration_unit": unidad,
            "main_genre": genero,
            "release_year": int(anio),
            "duration_value": int(duracion)
        }])

        X_input_encoded = self.preprocessor.transform(df_input)
        pred_age = float(self.modelo_multiple.predict(X_input_encoded)[0])
        pred_age_clamped = max(0.0, min(18.0, pred_age))

        # Determinar etiqueta recomendada
        if pred_age_clamped < 6:
            clasificacion = "TV-Y / G (Para todos los públicos)"
            badge = "TP"
            color = "#00c851"
            categoria = "Infantil / Familiar"
        elif pred_age_clamped < 10:
            clasificacion = "TV-PG / PG (+7 a +9 años)"
            badge = "+7"
            color = "#33b5e5"
            categoria = "Público Infantil Mayor"
        elif pred_age_clamped < 15:
            clasificacion = "PG-13 / TV-14 (+13 a +14 años)"
            badge = "+13"
            color = "#ffbb33"
            categoria = "Adolescentes y Adultos"
        elif pred_age_clamped < 17.5:
            clasificacion = "TV-MA / R (+16 a +17 años maduro)"
            badge = "+16"
            color = "#ff4444"
            categoria = "Audiencia Madura"
        else:
            clasificacion = "NC-17 / TV-MA (Exclusivo Adultos +18)"
            badge = "+18"
            color = "#e50914"
            categoria = "Solo Adultos"

        return {
            "edad_predicha_exacta": round(pred_age, 2),
            "edad_recomendada": round(pred_age_clamped, 1),
            "clasificacion_sugerida": clasificacion,
            "badge": badge,
            "color": color,
            "categoria": categoria,
            "parametros": {
                "tipo": tipo,
                "pais": pais,
                "genero": genero,
                "anio": anio,
                "duracion": duracion,
                "unidad": unidad
            }
        }

    def generar_grafico_recta(self, ruta_guardado=None):
        """Genera y guarda el gráfico 2D de la línea recta de regresión."""
        if not self.is_fitted:
            self.entrenar()

        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ruta_guardado = os.path.join(base_dir, "regresion_lineal_netflix.png")

        df = self.df_clean
        movies_mask = df["type"] == "Movie"
        X_movie_dur = df.loc[movies_mask, ["duration_value"]].values
        y_movie_age = df.loc[movies_mask, "target_age"].values

        m_pen = self.metricas["recta_2d"]["pendiente"]
        b_orig = self.metricas["recta_2d"]["intercepto"]

        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")

        dur_range = np.linspace(X_movie_dur.min(), X_movie_dur.max(), 200).reshape(-1, 1)
        age_pred_line = self.modelo_simple_duracion.predict(dur_range)

        # Muestreo representativo de puntos para generación instantánea del gráfico
        if len(X_movie_dur) > 1200:
            np.random.seed(42)
            indices = np.random.choice(len(X_movie_dur), size=1200, replace=False)
            x_plot = X_movie_dur[indices]
            y_plot = y_movie_age[indices]
        else:
            x_plot = X_movie_dur
            y_plot = y_movie_age

        ax.scatter(x_plot, y_plot, alpha=0.25, color="#00d4ff", s=14, label="Películas Reales (Muestra)")
        ax.plot(dur_range, age_pred_line, color="#e50914", linewidth=3,
                label=f"Regresión Lineal: y = {m_pen:.3f}x + {b_orig:.1f}")

        ax.set_title("Netflix: Duración (min) vs. Edad Recomendada", fontsize=13, fontweight="bold", color="#ffffff", pad=12)
        ax.set_xlabel("Duración de la Película (minutos)", fontsize=11, color="#dddddd")
        ax.set_ylabel("Edad Recomendada (Años)", fontsize=11, color="#dddddd")
        ax.set_ylim(-2, 22)
        ax.tick_params(colors="#aaaaaa")
        ax.legend(loc="upper left", facecolor="#141414", edgecolor="#333333", labelcolor="#ffffff")
        ax.grid(True, linestyle="--", alpha=0.25, color="#555555")

        plt.tight_layout()
        plt.savefig(ruta_guardado, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        # Copiar también al directorio raíz
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_raiz = os.path.join(parent_dir, "regresion_lineal_netflix.png")
        try:
            shutil.copyfile(ruta_guardado, ruta_raiz)
        except Exception:
            pass

        return ruta_guardado

if __name__ == "__main__":
    print("[*] Iniciando entrenamiento...", flush=True)
    pipeline = NetflixRegressionPipeline()
    metricas = pipeline.entrenar()
    print("[OK] Métricas del Modelo de Regresión:", flush=True)
    for k, v in metricas.items():
        print(f"  - {k}: {v}", flush=True)
    
    pred = pipeline.predecir(tipo="Movie", pais="United States", genero="Dramas", anio=2022, duracion=110)
    print("\n[OK] Ejemplo de Predicción:", flush=True)
    print(f"  - Edad recomendada: {pred['edad_recomendada']} años ({pred['clasificacion_sugerida']})", flush=True)

    img = pipeline.generar_grafico_recta()
    print(f"\n[OK] Gráfico guardado en: {img}", flush=True)
