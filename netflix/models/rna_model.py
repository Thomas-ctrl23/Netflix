# -*- coding: utf-8 -*-
"""
Model: RNAModel (Redes Neuronales Artificiales / Multi-Layer Perceptron)
Capa de Modelo para entrenamiento, evaluación y predicción con Red Neuronal Artificial (Scikit-Learn).
Predice la Edad Recomendada ('target_age') a partir de la Duración ('duration_value') usando MLPRegressor,
y la clasificación multivariable usando MLPClassifier.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from .limpieza_model import cargar_dataset, limpiar_datos

CATEGORICAL_FEATURES = ["type", "main_country", "duration_unit", "main_genre"]
NUMERIC_FEATURES = ["release_year", "duration_value"]

class NetflixRNAPipeline:
    def __init__(self, hidden_layers=(64, 32), max_iter=400):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.mlp_reg = None        # MLPRegressor 2D: Duración -> Edad
        self.mlp_clf = None        # MLPClassifier multivariable: rating
        self.scaler_x = None
        self.scaler_y = None
        self.preprocessor = None
        self.metricas = {}
        self.clases = []
        self.is_fitted = False
        self.df_clean = None

    def entrenar(self, df=None):
        if df is None:
            df_crudo = cargar_dataset()
            df, _ = limpiar_datos(df_crudo)

        self.df_clean = df.copy()

        # -------------------------------------------------------------
        # 1. ENTRENAMIENTO MODELO 2D (RNA: Duración -> Edad Recomendada)
        # -------------------------------------------------------------
        movies = self.df_clean[self.df_clean["type"] == "Movie"].copy()
        X_dur = movies[["duration_value"]].astype(float).values
        y_age = movies["target_age"].astype(float).values

        X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(
            X_dur, y_age, test_size=0.20, random_state=42
        )

        self.scaler_x = StandardScaler()
        X_train_dur_sc = self.scaler_x.fit_transform(X_train_dur)
        X_test_dur_sc = self.scaler_x.transform(X_test_dur)

        # Red Neuronal Multicapa (MLP) con regularización L2 y activación ReLU
        self.mlp_reg = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=32,
            learning_rate_init=0.01,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        )
        self.mlp_reg.fit(X_train_dur_sc, y_train_dur)

        y_test_pred_dur = self.mlp_reg.predict(X_test_dur_sc)
        r2_test_dur = float(r2_score(y_test_dur, y_test_pred_dur))
        mae_test_dur = float(mean_absolute_error(y_test_dur, y_test_pred_dur))

        # -------------------------------------------------------------
        # 2. ENTRENAMIENTO MODELO MULTIVARIABLE (MLPClassifier)
        # -------------------------------------------------------------
        top_clases = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG"]
        df_rna = self.df_clean[self.df_clean["rating"].isin(top_clases)].copy()

        X_multi = df_rna[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        y_multi = df_rna["rating"]

        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
            X_multi, y_multi, test_size=0.20, random_state=42, stratify=y_multi
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
                ("num", StandardScaler(), NUMERIC_FEATURES)
            ]
        )

        X_train_m_enc = self.preprocessor.fit_transform(X_train_m)
        X_test_m_enc = self.preprocessor.transform(X_test_m)

        self.mlp_clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=120,
            random_state=42
        )
        self.mlp_clf.fit(X_train_m_enc, y_train_m)

        y_test_pred_m = self.mlp_clf.predict(X_test_m_enc)
        self.clases = sorted(list(self.mlp_clf.classes_))

        self.metricas = {
            "n_train": int(len(X_train_dur)),
            "n_test": int(len(X_test_dur)),
            "r2_test": round(r2_test_dur, 4),
            "mae_test": round(mae_test_dur, 2),
            "capas_ocultas": list(self.hidden_layers),
            "arquitectura": f"1 -> {' -> '.join(map(str, self.hidden_layers))} -> 1",
            "activacion": "ReLU",
            "optimizador": "Adam",
            "epocas_iter": int(self.mlp_reg.n_iter_),
            "perdida_final": round(float(self.mlp_reg.loss_), 4),
            "accuracy_clf": round(float(accuracy_score(y_test_m, y_test_pred_m)), 4),
            "f1_clf": round(float(f1_score(y_test_m, y_test_pred_m, average="weighted", zero_division=0)), 4)
        }

        self.is_fitted = True
        return self.metricas

    def predecir_duracion(self, duracion_min):
        """Predice la edad recomendada según la duración usando la Red Neuronal Artificial (MLP)."""
        if not self.is_fitted:
            self.entrenar()

        dur = float(duracion_min)
        dur_sc = self.scaler_x.transform([[dur]])
        raw_pred = float(self.mlp_reg.predict(dur_sc)[0])
        pred_age = max(0.0, min(18.0, raw_pred))

        if pred_age < 6.0:
            clasificacion = "TV-Y / G (Todos los públicos / Infantil)"
            badge = "TP"
            color = "#00e676"
        elif pred_age < 11.5:
            clasificacion = "TV-Y7 / TV-PG (+7 años niños y familia)"
            badge = "+7"
            color = "#00d4ff"
        elif pred_age < 14.5:
            clasificacion = "PG-13 / TV-14 (+12 a +14 años adolescentes)"
            badge = "+13"
            color = "#ffb703"
        elif pred_age < 16.5:
            clasificacion = "TV-14 / TV-MA (+15 a +16 años)"
            badge = "+15"
            color = "#ff8c00"
        else:
            clasificacion = "TV-MA / R (+17 a +18 años adultos)"
            badge = "+18"
            color = "#e50914"

        arq = f"RNA MLP [{self.metricas.get('arquitectura', '1 -> 64 -> 32 -> 1')}]"
        detalle = f"{arq} | f({dur:.0f} min) = {pred_age:.2f} años (Pérdida Loss: {self.metricas.get('perdida_final', 0.05):.4f})"

        return {
            "duracion": dur,
            "edad_predicha_exacta": round(pred_age, 2),
            "edad_recomendada": round(pred_age, 1),
            "clasificacion_sugerida": clasificacion,
            "badge": badge,
            "color": color,
            "formula_detalle": detalle,
            "arquitectura": self.metricas.get("arquitectura", "1 -> 64 -> 32 -> 1"),
            "epocas": self.metricas.get("epocas_iter", 100),
            "loss": self.metricas.get("perdida_final", 0.05)
        }

    def generar_grafico_duracion_edad(self, ruta_guardado=None):
        """Genera el gráfico 2D de Duración vs Edad para la Red Neuronal Artificial (RNA)."""
        if not self.is_fitted:
            self.entrenar()

        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ruta_guardado = os.path.join(base_dir, "static", "rna_duracion_edad_netflix.png")

        movies = self.df_clean[self.df_clean["type"] == "Movie"].copy()
        X_dur = movies[["duration_value"]].astype(float).values
        y_age = movies["target_age"].astype(float).values

        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")

        dur_grid = np.linspace(X_dur.min(), X_dur.max(), 300).reshape(-1, 1)
        dur_grid_sc = self.scaler_x.transform(dur_grid)
        y_grid_pred = self.mlp_reg.predict(dur_grid_sc)

        # Muestra de datos reales
        if len(X_dur) > 1200:
            np.random.seed(42)
            idx = np.random.choice(len(X_dur), size=1200, replace=False)
            x_plot = X_dur[idx]
            y_plot = y_age[idx]
        else:
            x_plot = X_dur
            y_plot = y_age

        ax.scatter(x_plot, y_plot, alpha=0.25, color="#00d4ff", s=14, label="Películas Reales (Muestra)")

        # Curva aprendida por la Red Neuronal Artificial
        ax.plot(dur_grid, y_grid_pred, color="#00e676", linewidth=3.2,
                label=f"Ajuste RNA (MLP: {self.metricas.get('arquitectura', '1->64->32->1')})")

        ax.set_title("Netflix: Duración (min) vs. Edad Recomendada (Red Neuronal Artificial - RNA)",
                     fontsize=13, fontweight="bold", color="#ffffff", pad=12)
        ax.set_xlabel("Duración de la Película (minutos)", fontsize=11, color="#dddddd")
        ax.set_ylabel("Edad Recomendada (Años)", fontsize=11, color="#dddddd")
        ax.set_ylim(-2, 22)
        ax.tick_params(colors="#aaaaaa")
        ax.legend(loc="upper left", facecolor="#141414", edgecolor="#333333", labelcolor="#ffffff", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.25, color="#555555")

        plt.tight_layout()
        dir_name = os.path.dirname(ruta_guardado)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(ruta_guardado, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_static = os.path.join(parent_dir, "static", "rna_duracion_edad_netflix.png")
        if os.path.abspath(ruta_guardado) != os.path.abspath(ruta_static):
            try:
                shutil.copyfile(ruta_guardado, ruta_static)
            except Exception:
                pass

        return ruta_guardado

    def generar_grafico_perdida(self, ruta_guardado=None):
        """Genera el gráfico de la curva de convergencia/pérdida (Loss Curve) de la Red Neuronal."""
        if not self.is_fitted:
            self.entrenar()

        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ruta_guardado = os.path.join(base_dir, "static", "rna_curva_perdida_netflix.png")

        fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")

        loss_curve = self.mlp_reg.loss_curve_
        epocas = np.arange(1, len(loss_curve) + 1)

        ax.plot(epocas, loss_curve, color="#00e676", linewidth=2.5, label="Función de Pérdida (Loss - MSE)")
        ax.scatter([epocas[-1]], [loss_curve[-1]], color="#ffb703", s=70, zorder=5,
                   label=f"Mínimo alcanzado: {loss_curve[-1]:.4f}")

        ax.set_title("Curva de Aprendizaje / Pérdida de la Red Neuronal (RNA)",
                     fontsize=13, fontweight="bold", color="#ffffff", pad=12)
        ax.set_xlabel("Épocas / Iteraciones de Entrenamiento", fontsize=11, color="#dddddd")
        ax.set_ylabel("Pérdida Cuadrática Media (Loss)", fontsize=11, color="#dddddd")
        ax.tick_params(colors="#aaaaaa")
        ax.legend(loc="upper right", facecolor="#141414", edgecolor="#333333", labelcolor="#ffffff")
        ax.grid(True, linestyle="--", alpha=0.25, color="#555555")

        plt.tight_layout()
        dir_name = os.path.dirname(ruta_guardado)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(ruta_guardado, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_static = os.path.join(parent_dir, "static", "rna_curva_perdida_netflix.png")
        if os.path.abspath(ruta_guardado) != os.path.abspath(ruta_static):
            try:
                shutil.copyfile(ruta_guardado, ruta_static)
            except Exception:
                pass

        return ruta_guardado

    def generar_matriz_confusion(self, ruta_guardado=None):
        """Genera la matriz de confusión multiclase para la RNA clasificadora."""
        if not self.is_fitted:
            self.entrenar()

        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ruta_guardado = os.path.join(base_dir, "static", "matriz_confusion_rna_netflix.png")

        top_clases = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG"]
        df_rna = self.df_clean[self.df_clean["rating"].isin(top_clases)].copy()

        X = df_rna[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        y = df_rna["rating"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        X_test_encoded = self.preprocessor.transform(X_test)
        y_test_pred = self.mlp_clf.predict(X_test_encoded)

        mask_test = y_test.isin(top_clases)
        cm = confusion_matrix(y_test[mask_test], y_test_pred[mask_test], labels=top_clases)

        fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_clases)
        disp.plot(ax=ax, cmap="Greens", colorbar=True)

        ax.set_title("Matriz de Confusión - Red Neuronal Artificial (RNA)",
                     fontsize=13, fontweight="bold", color="#ffffff", pad=12)
        ax.set_xlabel("Clasificación Predicha por la Red", fontsize=11, color="#dddddd")
        ax.set_ylabel("Clasificación Real", fontsize=11, color="#dddddd")
        ax.tick_params(colors="#aaaaaa")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", color="#cccccc")
        plt.setp(ax.get_yticklabels(), color="#cccccc")

        plt.tight_layout()
        dir_mat = os.path.dirname(ruta_guardado)
        if dir_mat:
            os.makedirs(dir_mat, exist_ok=True)
        plt.savefig(ruta_guardado, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_static = os.path.join(parent_dir, "static", "matriz_confusion_rna_netflix.png")
        if os.path.abspath(ruta_guardado) != os.path.abspath(ruta_static):
            try:
                shutil.copyfile(ruta_guardado, ruta_static)
            except Exception:
                pass

        return ruta_guardado

if __name__ == "__main__":
    print("[*] Entrenando Pipeline de RNA para Netflix...", flush=True)
    pipeline = NetflixRNAPipeline()
    metricas = pipeline.entrenar()
    print("\n[OK] Métricas de la Red Neuronal:", flush=True)
    for k, v in metricas.items():
        print(f"  - {k}: {v}", flush=True)

    pred = pipeline.predecir_duracion(105)
    print(f"\n[OK] Predicción RNA (105 min): {pred['edad_recomendada']} años | {pred['clasificacion_sugerida']}", flush=True)
    print(f"  - Detalle: {pred['formula_detalle']}", flush=True)

    img_2d = pipeline.generar_grafico_duracion_edad()
    img_loss = pipeline.generar_grafico_perdida()
    print(f"\n[OK] Gráficos guardados en:\n  - {img_2d}\n  - {img_loss}", flush=True)
