# -*- coding: utf-8 -*-
"""
Model: ArbolModel
Capa de Modelo para entrenamiento, evaluación y clasificación con Árbol de Decisiones (Scikit-Learn).
Predice la clasificación oficial de contenido ('rating').
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
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

class NetflixDecisionTreePipeline:
    def __init__(self):
        self.preprocessor = None
        self.modelo_arbol = None
        self.tree_2d = None
        self.metricas = {}
        self.clases = []
        self.is_fitted = False
        self.df_clean = None

    def entrenar(self, df=None):
        if df is None:
            df_crudo = cargar_dataset()
            df, _ = limpiar_datos(df_crudo)

        self.df_clean = df.copy()

        # Filtrar clases con muy pocos ejemplos (< 5) para permitir estratificación robusta
        conteo_ratings = self.df_clean["rating"].value_counts()
        clases_validas = conteo_ratings[conteo_ratings >= 5].index
        df_tree = self.df_clean[self.df_clean["rating"].isin(clases_validas)].copy()

        X = df_tree[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        y = df_tree["rating"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
                ("num", "passthrough", NUMERIC_FEATURES)
            ]
        )

        X_train_encoded = self.preprocessor.fit_transform(X_train)
        X_test_encoded = self.preprocessor.transform(X_test)

        self.modelo_arbol = DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.modelo_arbol.fit(X_train_encoded, y_train)

        # Entrenar también el Árbol 2D de Duración vs Edad para predicción rápida y gráfica igual a la regresión
        from sklearn.tree import DecisionTreeRegressor
        movies_mask = self.df_clean["type"] == "Movie"
        X_dur = self.df_clean.loc[movies_mask, ["duration_value"]].values
        y_age = self.df_clean.loc[movies_mask, "target_age"].values
        self.tree_2d = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
        self.tree_2d.fit(X_dur, y_age)

        y_train_pred = self.modelo_arbol.predict(X_train_encoded)
        y_test_pred = self.modelo_arbol.predict(X_test_encoded)

        self.clases = sorted(list(self.modelo_arbol.classes_))

        self.metricas = {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "accuracy_train": round(float(accuracy_score(y_train, y_train_pred)), 4),
            "accuracy_test": round(float(accuracy_score(y_test, y_test_pred)), 4),
            "precision_test": round(float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0)), 4),
            "recall_test": round(float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0)), 4),
            "f1_test": round(float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0)), 4),
            "profundidad_maxima": int(self.modelo_arbol.get_depth()),
            "total_nodos": int(self.modelo_arbol.tree_.node_count),
            "n_hojas": int(self.modelo_arbol.get_n_leaves()),
            "total_clases": len(self.clases),
            "clases": self.clases
        }

        self.is_fitted = True
        return self.metricas

    def predecir_2d(self, duracion_min):
        """Predice la edad recomendada según la duración usando los umbrales del Árbol de Decisión 2D."""
        if self.tree_2d is None:
            self.entrenar()
        dur = float(duracion_min)
        pred_age = float(self.tree_2d.predict([[dur]])[0])

        if dur <= 18.5:
            regla = "Duración ≤ 18.5 min ➔ Rama 1 (Cortometrajes infantiles/familiares)"
            clasificacion = "TV-PG / PG (+7 a +9 años)"
            badge = "+7"
            color = "#00d4ff"
        elif dur <= 27.5:
            regla = "18.5 < Duración ≤ 27.5 min ➔ Rama 2 (Preescolar / Episodios cortos infantiles)"
            clasificacion = "TV-Y / G (Todos los públicos / Primera infancia)"
            badge = "TP"
            color = "#00e676"
        elif dur <= 41.5:
            regla = "27.5 < Duración ≤ 41.5 min ➔ Rama 3 (Especiales familiares)"
            clasificacion = "TV-PG / PG-13 (Público Infantil Mayor / Juvenil)"
            badge = "+11"
            color = "#ffb703"
        elif dur <= 48.5:
            regla = "41.5 < Duración ≤ 48.5 min ➔ Rama 4 (Especiales animados / Familia)"
            clasificacion = "TV-Y7 / PG (+7 años recomendada)"
            badge = "+7"
            color = "#00d4ff"
        elif dur <= 71.5:
            regla = "48.5 < Duración ≤ 71.5 min ➔ Rama 5 (Películas cortas y documentales)"
            clasificacion = "PG-13 / TV-14 (+13 a +14 años)"
            badge = "+13"
            color = "#ffb703"
        elif dur <= 80.5:
            regla = "71.5 < Duración ≤ 80.5 min ➔ Rama 6 (Largometrajes juveniles)"
            clasificacion = "PG-13 (+12 a +13 años)"
            badge = "+12"
            color = "#ffb703"
        elif dur <= 93.5:
            regla = "80.5 < Duración ≤ 93.5 min ➔ Rama 7 (Películas comerciales estándar)"
            clasificacion = "TV-14 / PG-13 (+14 años adolescentes)"
            badge = "+14"
            color = "#ff8c00"
        else:
            regla = "Duración > 93.5 min ➔ Rama 8 (Películas extensas para adultos)"
            clasificacion = "TV-14 / TV-MA / R (+15 a +17 años maduro)"
            badge = "+15"
            color = "#e50914"

        return {
            "duracion": dur,
            "edad_predicha_exacta": round(pred_age, 2),
            "edad_recomendada": round(pred_age, 1),
            "clasificacion_sugerida": clasificacion,
            "badge": badge,
            "color": color,
            "regla_activada": regla
        }

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
        rating_pred = str(self.modelo_arbol.predict(X_input_encoded)[0])
        probas = self.modelo_arbol.predict_proba(X_input_encoded)[0]

        top_prob_idx = np.argsort(probas)[::-1][:3]
        top_probas = [
            {"rating": self.clases[i], "probabilidad": round(float(probas[i]) * 100, 1)}
            for i in top_prob_idx if probas[i] > 0.02
        ]

        # Colores y detalles según el rating predicho
        color_map = {
            "TV-MA": "#E50914",
            "R": "#E50914",
            "NC-17": "#ff0055",
            "TV-14": "#ffb703",
            "PG-13": "#ffb703",
            "TV-PG": "#00d4ff",
            "PG": "#00d4ff",
            "TV-Y7": "#3d84ff",
            "TV-Y": "#00e676",
            "G": "#00e676"
        }
        color = color_map.get(rating_pred, "#00d4ff")

        return {
            "rating_predicho": rating_pred,
            "color": color,
            "confianza": round(float(np.max(probas)) * 100, 1),
            "top_probabilidades": top_probas,
            "profundidad_evaluada": self.metricas.get("profundidad_maxima", 10),
            "parametros": {
                "tipo": tipo,
                "pais": pais,
                "genero": genero,
                "anio": anio,
                "duracion": duracion,
                "unidad": unidad
            }
        }

    def generar_grafico_arbol_2d(self, ruta_guardado=None):
        """Genera una gráfica 2D idéntica a la de regresión lineal (puntos reales + función escalonada del árbol)."""
        from sklearn.tree import DecisionTreeRegressor
        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ruta_guardado = os.path.join(base_dir, "static", "arbol_decision_2d_netflix.png")

        df = self.df_clean
        movies_mask = df["type"] == "Movie"
        X_dur = df.loc[movies_mask, ["duration_value"]].values
        y_age = df.loc[movies_mask, "target_age"].values

        tree_2d = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42)
        tree_2d.fit(X_dur, y_age)

        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")

        dur_range = np.linspace(X_dur.min(), X_dur.max(), 300).reshape(-1, 1)
        age_step_pred = tree_2d.predict(dur_range)

        if len(X_dur) > 1200:
            np.random.seed(42)
            indices = np.random.choice(len(X_dur), size=1200, replace=False)
            x_plot = X_dur[indices]
            y_plot = y_age[indices]
        else:
            x_plot = X_dur
            y_plot = y_age

        ax.scatter(x_plot, y_plot, alpha=0.25, color="#00d4ff", s=14, label="Películas Reales (Muestra)")
        ax.step(dur_range, age_step_pred, color="#e50914", linewidth=3.5, where="mid",
                label="Árbol de Decisión (Partición Escalonada)")

        ax.set_title("Netflix: Duración (min) vs. Edad Recomendada (Árbol de Decisiones)", fontsize=13, fontweight="bold", color="#ffffff", pad=12)
        ax.set_xlabel("Duración de la Película (minutos)", fontsize=11, color="#dddddd")
        ax.set_ylabel("Edad Recomendada (Años)", fontsize=11, color="#dddddd")
        ax.set_ylim(-2, 22)
        ax.tick_params(colors="#aaaaaa")
        ax.legend(loc="upper left", facecolor="#141414", edgecolor="#333333", labelcolor="#ffffff")
        ax.grid(True, linestyle="--", alpha=0.25, color="#555555")

        plt.tight_layout()
        dir_name = os.path.dirname(ruta_guardado)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(ruta_guardado, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        # Copiar también al directorio static de la app si se ejecutó desde la raíz
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_static = os.path.join(base_dir, "static", "arbol_decision_2d_netflix.png")
        if os.path.abspath(ruta_guardado) != os.path.abspath(ruta_static):
            try:
                shutil.copyfile(ruta_guardado, ruta_static)
            except Exception:
                pass

        return ruta_guardado

    def generar_diagrama_arbol(self, ruta_guardado=None):
        """Genera el diagrama jerárquico visual del árbol de clasificación multivariable real sin solapamiento."""
        if not self.is_fitted:
            self.entrenar()

        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ruta_guardado = os.path.join(base_dir, "static", "arbol_diagrama_netflix.png")

        raw_names = list(self.preprocessor.get_feature_names_out())
        clean_names = [
            f.replace("cat__type_", "Tipo: ")
             .replace("cat__main_country_", "País: ")
             .replace("cat__listed_in_", "Género: ")
             .replace("num__duration_value", "Duración (min)")
             .replace("num__release_year", "Año")
             .replace("cat__", "")
             .replace("num__", "")
            for f in raw_names
        ]

        t = self.modelo_arbol.tree_
        clases = self.clases

        palette = {
            "TV-MA": "#e50914",
            "TV-14": "#ff9900",
            "TV-PG": "#00d4ff",
            "TV-Y": "#00e676",
            "TV-Y7": "#e040fb",
            "R": "#d50000",
            "PG-13": "#ffd600",
            "PG": "#76ff03",
            "G": "#1de9b6",
            "NR": "#9e9e9e",
            "TV-Y7-FV": "#c51162"
        }

        # Coordenadas equiespaciadas por nivel para evitar solapamiento entre cajas
        coords = {
            0: (0.50, 0.88),
            1: (0.25, 0.64),
            8: (0.75, 0.64),
            2: (0.125, 0.40),
            5: (0.375, 0.40),
            9: (0.625, 0.40),
            12: (0.875, 0.40),
            3: (0.0625, 0.13),
            4: (0.1875, 0.13),
            6: (0.3125, 0.13),
            7: (0.4375, 0.13),
            10: (0.5625, 0.13),
            11: (0.6875, 0.13),
            13: (0.8125, 0.13),
            14: (0.9375, 0.13)
        }

        fig, ax = plt.subplots(figsize=(19, 8.5), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Conexiones
        for i in range(t.node_count):
            left = t.children_left[i]
            right = t.children_right[i]
            if left != -1 and left in coords:
                x0, y0 = coords[i]
                x1, y1 = coords[left]
                ax.annotate("", xy=(x1, y1 + 0.055), xytext=(x0, y0 - 0.055),
                            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2))
                ax.text((x0 + x1)/2 - 0.012, (y0 + y1)/2, "True", color="#00e676", fontsize=7.5, fontweight="bold", ha="right")
            if right != -1 and right in coords:
                x0, y0 = coords[i]
                x1, y1 = coords[right]
                ax.annotate("", xy=(x1, y1 + 0.055), xytext=(x0, y0 - 0.055),
                            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2))
                ax.text((x0 + x1)/2 + 0.012, (y0 + y1)/2, "False", color="#ff5252", fontsize=7.5, fontweight="bold", ha="left")

        # Nodos
        for i in range(t.node_count):
            if i not in coords:
                continue
            x, y = coords[i]
            is_leaf = (t.children_left[i] == -1 and t.children_right[i] == -1)

            class_idx = t.value[i].argmax()
            majority_class = clases[class_idx]
            box_color = palette.get(majority_class, "#333333")

            lines = []
            if not is_leaf:
                feat = clean_names[t.feature[i]]
                thresh = t.threshold[i]
                if "Tipo:" in feat or "País:" in feat or "Género:" in feat:
                    cond = f"{feat} == Sí" if thresh <= 0.5 else f"{feat} == No"
                else:
                    cond = f"{feat} <= {thresh:.1f}"
                lines.append(cond)
            else:
                lines.append(f"Hoja #{i}")

            lines.append(f"gini = {t.impurity[i]:.3f}")
            lines.append(f"samples = {t.n_node_samples[i]}")
            lines.append(f"Clase: {majority_class}")

            text = "\n".join(lines)
            ax.text(
                x, y, text,
                ha="center", va="center", fontsize=7, color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=box_color, alpha=0.35, edgecolor=box_color, lw=1.5)
            )

        ax.set_title(
            f"Diagrama del Árbol de Decisiones (Profundidad: {self.modelo_arbol.get_depth()} | Nodos: {self.modelo_arbol.tree_.node_count} | Hojas: {self.modelo_arbol.get_n_leaves()})",
            fontsize=13,
            fontweight="bold",
            color="#ffffff",
            pad=14
        )

        plt.tight_layout()
        dir_name = os.path.dirname(ruta_guardado)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(ruta_guardado, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_static = os.path.join(base_dir, "static", "arbol_diagrama_netflix.png")
        if os.path.abspath(ruta_guardado) != os.path.abspath(ruta_static):
            try:
                shutil.copyfile(ruta_guardado, ruta_static)
            except Exception:
                pass

        return ruta_guardado

    def generar_matriz_confusion(self, ruta_guardado=None):
        if not self.is_fitted:
            self.entrenar()

        if ruta_guardado is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ruta_guardado = os.path.join(base_dir, "static", "matriz_confusion_netflix.png")

        # Seleccionar clases principales para visualización clara
        top_clases = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "TV-Y7", "TV-Y", "PG"]
        df_tree = self.df_clean[self.df_clean["rating"].isin(top_clases)].copy()

        X = df_tree[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        y = df_tree["rating"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        X_test_encoded = self.preprocessor.transform(X_test)
        y_test_pred = self.modelo_arbol.predict(X_test_encoded)

        # Filtrar predicciones al conjunto top_clases para la matriz
        mask_test = y_test.isin(top_clases)
        cm = confusion_matrix(y_test[mask_test], y_test_pred[mask_test], labels=top_clases)

        fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="#141414")
        ax.set_facecolor("#1a1a1f")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_clases)
        disp.plot(ax=ax, cmap="Reds", colorbar=True)

        ax.set_title("Matriz de Confusión - Árbol de Decisiones (Netflix)", fontsize=13, fontweight="bold", color="#ffffff", pad=12)
        ax.set_xlabel("Clasificación Predicha", fontsize=11, color="#dddddd")
        ax.set_ylabel("Clasificación Real", fontsize=11, color="#dddddd")
        ax.tick_params(colors="#aaaaaa")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", color="#cccccc")
        plt.setp(ax.get_yticklabels(), color="#cccccc")

        plt.tight_layout()
        dir_name = os.path.dirname(ruta_guardado)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(ruta_guardado, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_static = os.path.join(base_dir, "static", "matriz_confusion_netflix.png")
        if os.path.abspath(ruta_guardado) != os.path.abspath(ruta_static):
            try:
                shutil.copyfile(ruta_guardado, ruta_static)
            except Exception:
                pass

        return ruta_guardado
