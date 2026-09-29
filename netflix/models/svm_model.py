# -*- coding: utf-8 -*-
"""
Model: SVMModel (Máquina de Vectores de Soporte)
Capa de Modelo para entrenamiento, evaluación y clasificación con Support Vector Machine (Scikit-Learn).
Predice la clasificación oficial de contenido ('rating') mediante hiperplanos óptimos de separación y kernel RBF.
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
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from .limpieza_model import cargar_dataset, limpiar_datos

CATEGORICAL_FEATURES = ["type", "main_country", "duration_unit", "main_genre"]
NUMERIC_FEATURES = ["release_year", "duration_value"]

class NetflixSVMPipeline:
    def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.preprocessor = None
        self.modelo_svm = None
        self.modelo_2d = None
        self.scaler_2d = None
        self.svr_dur = None
        self.scaler_dur = None
        self.metricas = {}
        self.clases = []
        self.is_fitted = False
        self.df_clean = None

    def entrenar(self, df=None):
        if df is None:
            df_crudo = cargar_dataset()
            df, _ = limpiar_datos(df_crudo)

        self.df_clean = df.copy()

        # Filtrar clases con al menos 10 registros para estratificación confiable
        conteo_ratings = self.df_clean["rating"].value_counts()
        clases_validas = conteo_ratings[conteo_ratings >= 10].index
        df_svm = self.df_clean[self.df_clean["rating"].isin(clases_validas)].copy()

        X = df_svm[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        y = df_svm["rating"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
                ("num", StandardScaler(), NUMERIC_FEATURES)
            ]
        )

        X_train_encoded = self.preprocessor.fit_transform(X_train)
        X_test_encoded = self.preprocessor.transform(X_test)

        self.modelo_svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=False,
            random_state=42
        )
        self.modelo_svm.fit(X_train_encoded, y_train)

        y_train_pred = self.modelo_svm.predict(X_train_encoded)
        y_test_pred = self.modelo_svm.predict(X_test_encoded)

        self.clases = sorted(list(self.modelo_svm.classes_))
        total_sv = int(np.sum(self.modelo_svm.n_support_))
        sv_dict = {cls: int(n) for cls, n in zip(self.modelo_svm.classes_, self.modelo_svm.n_support_)}

        # Entrenar modelo 2D (Películas: Duración vs Año para separar Adulto vs Familiar/General)
        movies = self.df_clean[self.df_clean["type"] == "Movie"].copy()
        movies["is_mature"] = movies["rating"].isin(["TV-MA", "R", "NC-17"]).astype(int)
        
        sample_size = min(1500, len(movies))
        movies_sample = movies.sample(n=sample_size, random_state=42)
        X_2d_raw = movies_sample[["duration_value", "release_year"]].astype(float).values
        y_2d = movies_sample["is_mature"].astype(int).values

        self.scaler_2d = StandardScaler()
        X_2d_scaled = self.scaler_2d.fit_transform(X_2d_raw)

        self.modelo_2d = SVC(kernel="rbf", C=1.0, random_state=42)
        self.modelo_2d.fit(X_2d_scaled, y_2d)
        acc_2d = float(self.modelo_2d.score(X_2d_scaled, y_2d))

        # Entrenar Regresor SVR para estimar Edad Recomendada a partir de Duración (Duración vs Edad)
        from sklearn.svm import SVR
        X_dur = movies[["duration_value"]].astype(float).values
        y_age = movies["target_age"].astype(float).values

        self.scaler_dur = StandardScaler()
        X_dur_scaled = self.scaler_dur.fit_transform(X_dur)

        self.svr_dur = SVR(kernel="rbf", C=5.0, epsilon=0.8)
        self.svr_dur.fit(X_dur_scaled, y_age)

        self.metricas = {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "accuracy_train": round(float(accuracy_score(y_train, y_train_pred)), 4),
            "accuracy_test": round(float(accuracy_score(y_test, y_test_pred)), 4),
            "precision_test": round(float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0)), 4),
            "recall_test": round(float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0)), 4),
            "f1_test": round(float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0)), 4),
            "kernel": self.kernel.upper(),
            "param_c": self.C,
            "param_gamma": self.gamma,
            "total_vectores_soporte": total_sv,
            "vectores_soporte_por_clase": sv_dict,
            "total_clases": len(self.clases),
            "clases": self.clases,
            "accuracy_2d": round(acc_2d, 4),
            "total_sv_2d": int(len(self.modelo_2d.support_vectors_)),
            "total_sv_dur": int(len(self.svr_dur.support_vectors_))
        }

        self.is_fitted = True
        return self.metricas

    def predecir(self, tipo="Movie", pais="United States", genero="Dramas", anio=2021, duracion=95):
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
        rating_pred = str(self.modelo_svm.predict(X_input_encoded)[0])

        # Obtener valores de la función de decisión y aplicar softmax para obtener probabilidades
        dfunc = self.modelo_svm.decision_function(X_input_encoded)[0]
        e_x = np.exp(dfunc - np.max(dfunc))
        probas = e_x / np.sum(e_x)

        top_prob_idx = np.argsort(probas)[::-1][:3]
        top_probas = [
            {"rating": self.clases[i], "probabilidad": round(float(probas[i]) * 100, 1)}
            for i in top_prob_idx if probas[i] > 0.01
        ]

        confianza = round(float(np.max(probas)) * 100, 1)

        # Distancia al hiperplano separador (margen del punto)
        distancia_hiperplano = round(float(np.max(dfunc)), 3)

        color_map = {
            "TV-MA": "#E50914",
            "R": "#E50914",
            "NC-17": "#ff0055",
            "TV-14": "#ffb703",
            "PG-13": "#ffb703",
            "TV-PG": "#00d4ff",
            "PG": "#00d4ff",
            "TV-Y7": "#a855f7",
            "TV-Y": "#00e676",
            "G": "#00e676"
        }
        color = color_map.get(rating_pred, "#a855f7")

        return {
            "rating_predicho": rating_pred,
            "color": color,
            "confianza": confianza,
            "distancia_hiperplano": distancia_hiperplano,
            "top_probabilidades": top_probas,
            "kernel": self.kernel.upper(),
            "total_vectores_soporte": self.metricas.get("total_vectores_soporte", 0),
            "parametros": {
                "tipo": tipo,
                "pais": pais,
                "genero": genero,
                "anio": anio,
                "duracion": duracion,
                "unidad": unidad
            }
        }

    def predecir_2d(self, duracion=95, anio=2021):
        if not self.is_fitted:
            self.entrenar()

        val_raw = np.array([[float(duracion), float(anio)]])
        val_scaled = self.scaler_2d.transform(val_raw)
        pred_bin = int(self.modelo_2d.predict(val_scaled)[0])
        dist = float(self.modelo_2d.decision_function(val_scaled)[0])

        if pred_bin == 1:
            categoria = "Contenido Maduro / Adulto (+17 / TV-MA / R)"
            color = "#E50914"
            badge = "+18"
            zona = "Zona Hiperplano Positiva (Maduro)"
        else:
            categoria = "Contenido General / Familiar (TV-PG / TV-14 / TV-Y / PG)"
            color = "#00d4ff"
            badge = "TP/+13"
            zona = "Zona Hiperplano Negativa (Familiar)"

        # Margen funcional
        if abs(dist) < 0.3:
            margen_info = "Punto en el Margen Crítico (Vector de Soporte Potencial)"
        elif dist > 0:
            margen_info = f"Margen hacia Maduro: +{abs(dist):.2f}"
        else:
            margen_info = f"Margen hacia Familiar: -{abs(dist):.2f}"

        return {
            "duracion": duracion,
            "anio": anio,
            "categoria": categoria,
            "color": color,
            "badge": badge,
            "distancia_hiperplano": round(dist, 3),
            "zona": zona,
            "margen_info": margen_info
        }

    def predecir_duracion(self, duracion_min):
        """Predice la edad recomendada según la duración en minutos usando Support Vector Regression (SVR)."""
        if self.svr_dur is None:
            self.entrenar()

        dur = float(duracion_min)
        dur_scaled = self.scaler_dur.transform([[dur]])
        raw_pred = float(self.svr_dur.predict(dur_scaled)[0])
        pred_age = max(0.0, min(18.0, raw_pred))

        if pred_age < 6.0:
            clasificacion = "TV-Y / G (Todos los públicos / Infantil)"
            badge = "TP"
            color = "#00e676"
        elif pred_age < 10.0:
            clasificacion = "TV-PG / PG (+7 a +9 años)"
            badge = "+7"
            color = "#00d4ff"
        elif pred_age < 14.5:
            clasificacion = "PG-13 / TV-14 (+13 a +14 años)"
            badge = "+13"
            color = "#ffb703"
        elif pred_age < 17.0:
            clasificacion = "TV-14 / TV-MA (+15 a +16 años)"
            badge = "+15"
            color = "#ff8c00"
        else:
            clasificacion = "TV-MA / R / NC-17 (+17 a +18 años maduro)"
            badge = "+18"
            color = "#e50914"

        return {
            "duracion": dur,
            "edad_predicha_exacta": round(pred_age, 2),
            "edad_recomendada": round(pred_age, 1),
            "clasificacion_sugerida": clasificacion,
            "badge": badge,
            "color": color,
            "formula_detalle": f"SVR RBF: f({dur:.0f} min) = {pred_age:.2f} años (Margen ε = 0.8)"
        }

    def generar_grafico_duracion_edad(self, ruta_guardado=None):
        """Genera el gráfico 2D de Duración vs Edad para SVM (análogo al de regresión y árbol)."""
        if not self.is_fitted:
            self.entrenar()

        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ruta_guardado = os.path.join(base_dir, "static", "svm_duracion_edad_netflix.png")

        movies = self.df_clean[self.df_clean["type"] == "Movie"].copy()
        X_dur = movies[["duration_value"]].astype(float).values
        y_age = movies["target_age"].astype(float).values

        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")

        dur_grid = np.linspace(X_dur.min(), X_dur.max(), 300).reshape(-1, 1)
        dur_grid_scaled = self.scaler_dur.transform(dur_grid)
        y_grid_pred = self.svr_dur.predict(dur_grid_scaled)

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

        # Curva de regresión SVR
        ax.plot(dur_grid, y_grid_pred, color="#a855f7", linewidth=3.2,
                label="Ajuste SVM (Support Vector Regression)")

        # Tubo de margen épsilon
        ax.plot(dur_grid, y_grid_pred + 0.8, color="#ffb703", linestyle="--", linewidth=1.5,
                label="Margen Superior (f(x) + ε)")
        ax.plot(dur_grid, y_grid_pred - 0.8, color="#ffb703", linestyle="--", linewidth=1.5,
                label="Margen Inferior (f(x) - ε)")

        ax.set_title("Netflix: Duración (min) vs. Edad Recomendada (SVM - SVR)", fontsize=13, fontweight="bold", color="#ffffff", pad=12)
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
        ruta_raiz = os.path.join(parent_dir, "svm_duracion_edad_netflix.png")
        if os.path.abspath(ruta_guardado) != os.path.abspath(ruta_raiz):
            try:
                shutil.copyfile(ruta_guardado, ruta_raiz)
            except Exception:
                pass

        return ruta_guardado

    def generar_grafico_hiperplano(self, ruta_guardado=None):
        """
        Genera la visualización 2D del Hiperplano de Separación, márgenes y Vectores de Soporte
        para la frontera entre contenido Maduro vs Familiar en función de la duración y año.
        """
        if not self.is_fitted:
            self.entrenar()

        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ruta_guardado = os.path.join(base_dir, "static", "svm_hiperplano_netflix.png")

        movies = self.df_clean[self.df_clean["type"] == "Movie"].copy()
        movies["is_mature"] = movies["rating"].isin(["TV-MA", "R", "NC-17"]).astype(int)

        sample_size = min(900, len(movies))
        movies_sample = movies.sample(n=sample_size, random_state=42)
        X_raw = movies_sample[["duration_value", "release_year"]].astype(float).values
        y = movies_sample["is_mature"].astype(int).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        clf = SVC(kernel="rbf", C=1.0, random_state=42)
        clf.fit(X_scaled, y)

        fig, ax = plt.subplots(figsize=(9.5, 6), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")

        # Crear malla para la frontera de decisión
        x_min, x_max = float(X_raw[:, 0].min()) - 5.0, float(X_raw[:, 0].max()) + 5.0
        y_min, y_max = float(X_raw[:, 1].min()) - 2.0, float(X_raw[:, 1].max()) + 2.0

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
        Z = clf.decision_function(grid_scaled).reshape(xx.shape)

        # Región coloreada de decisión
        ax.contourf(xx, yy, Z, levels=[-100, 0, 100], alpha=0.25, colors=["#00d4ff", "#e50914"])

        # Hiperplano (Z = 0) y Márgenes (Z = -1, Z = +1)
        ax.contour(
            xx, yy, Z,
            levels=[-1.0, 0.0, 1.0],
            linestyles=["--", "-", "--"],
            colors=["#ffb703", "#ffffff", "#ffb703"],
            linewidths=[1.8, 3.2, 1.8]
        )

        # Dibujar puntos de datos reales
        mask_mature = y == 1
        mask_fam = y == 0

        ax.scatter(
            X_raw[mask_fam, 0], X_raw[mask_fam, 1],
            c="#00d4ff", s=20, alpha=0.55, edgecolors="none",
            label="Contenido Familiar / Menores (TV-PG / TV-14 / TV-Y)"
        )
        ax.scatter(
            X_raw[mask_mature, 0], X_raw[mask_mature, 1],
            c="#e50914", s=20, alpha=0.55, edgecolors="none",
            label="Contenido Adulto / Maduro (TV-MA / R)"
        )

        # Resaltar Vectores de Soporte con aros dorados/púrpuras
        sv_scaled = clf.support_vectors_
        sv_raw = scaler.inverse_transform(sv_scaled)

        # Mostrar muestra de vectores de soporte para nitidez visual
        if len(sv_raw) > 250:
            idx_sv = np.random.choice(len(sv_raw), size=250, replace=False)
            sv_plot = sv_raw[idx_sv]
        else:
            sv_plot = sv_raw

        ax.scatter(
            sv_plot[:, 0], sv_plot[:, 1],
            s=70, linewidth=1.5, facecolors="none", edgecolors="#a855f7",
            label=f"Vectores de Soporte (Total: {len(sv_raw)})"
        )

        ax.set_title(
            f"Máquina de Vectores de Soporte (SVM): Hiperplano Óptimo y Vectores de Soporte",
            fontsize=13, fontweight="bold", color="#ffffff", pad=12
        )
        ax.set_xlabel("Duración de la Película (minutos)", fontsize=11, color="#dddddd")
        ax.set_ylabel("Año de Lanzamiento (release_year)", fontsize=11, color="#dddddd")
        ax.set_xlim(30, 210)
        ax.set_ylim(max(1975, y_min), min(2023, y_max))
        ax.tick_params(colors="#aaaaaa")
        ax.grid(True, linestyle="--", alpha=0.25, color="#555555")

        # Leyenda estilizada
        ax.legend(
            loc="upper left",
            facecolor="#141414",
            edgecolor="#333333",
            labelcolor="#ffffff",
            fontsize=8.5
        )

        plt.tight_layout()
        dir_hip = os.path.dirname(ruta_guardado)
        if dir_hip:
            os.makedirs(dir_hip, exist_ok=True)
        plt.savefig(ruta_guardado, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        # Copiar también al directorio raíz si aplica
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_raiz = os.path.join(parent_dir, "svm_hiperplano_netflix.png")
        if os.path.abspath(ruta_guardado) != os.path.abspath(ruta_raiz):
            try:
                shutil.copyfile(ruta_guardado, ruta_raiz)
            except Exception:
                pass

        return ruta_guardado

    def generar_matriz_confusion(self, ruta_guardado=None):
        if not self.is_fitted:
            self.entrenar()

        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ruta_guardado = os.path.join(base_dir, "static", "matriz_confusion_svm_netflix.png")

        top_clases = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG"]
        df_svm = self.df_clean[self.df_clean["rating"].isin(top_clases)].copy()

        X = df_svm[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        y = df_svm["rating"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        X_test_encoded = self.preprocessor.transform(X_test)
        y_test_pred = self.modelo_svm.predict(X_test_encoded)

        mask_test = y_test.isin(top_clases)
        cm = confusion_matrix(y_test[mask_test], y_test_pred[mask_test], labels=top_clases)

        fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_clases)
        disp.plot(ax=ax, cmap="Purples", colorbar=True)

        ax.set_title("Matriz de Confusión - SVM (Support Vector Machine)", fontsize=13, fontweight="bold", color="#ffffff", pad=12)
        ax.set_xlabel("Clasificación Predicha por SVM", fontsize=11, color="#dddddd")
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
        ruta_raiz = os.path.join(parent_dir, "matriz_confusion_svm_netflix.png")
        if os.path.abspath(ruta_guardado) != os.path.abspath(ruta_raiz):
            try:
                shutil.copyfile(ruta_guardado, ruta_raiz)
            except Exception:
                pass

        return ruta_guardado

if __name__ == "__main__":
    print("[*] Entrenando Pipeline de SVM para Netflix...", flush=True)
    pipeline = NetflixSVMPipeline()
    metricas = pipeline.entrenar()
    print("\n[OK] Métricas de la Máquina de Vectores de Soporte:", flush=True)
    for k, v in metricas.items():
        if k != "vectores_soporte_por_clase":
            print(f"  - {k}: {v}", flush=True)

    pred = pipeline.predecir(tipo="Movie", pais="United States", genero="Action & Adventure", anio=2021, duracion=115)
    print("\n[OK] Ejemplo de Predicción Multivariable:", flush=True)
    print(f"  - Rating Predicho: {pred['rating_predicho']} (Confianza: {pred['confianza']}%)", flush=True)

    pred_2d = pipeline.predecir_2d(duracion=120, anio=2020)
    print(f"  - Predicción 2D: {pred_2d['categoria']} (Distancia al hiperplano: {pred_2d['distancia_hiperplano']})", flush=True)

    img_hiper = pipeline.generar_grafico_hiperplano()
    img_matriz = pipeline.generar_matriz_confusion()
    print(f"\n[OK] Gráficos guardados en:\n  - {img_hiper}\n  - {img_matriz}", flush=True)
