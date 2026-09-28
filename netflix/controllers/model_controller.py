# -*- coding: utf-8 -*-
"""
Controller: ModelController
Controlador central que gestiona la selección de modelos, las rutas de visualización
y las llamadas a los modelos de Regresión Lineal y Árbol de Decisiones.
"""

import os
from flask import Blueprint, render_template, request, jsonify, send_file
try:
    from models import (
        cargar_dataset,
        limpiar_datos,
        guardar_datos_limpios,
        NetflixRegressionPipeline,
        NetflixDecisionTreePipeline
    )
except (ImportError, ValueError):
    from ..models import (
        cargar_dataset,
        limpiar_datos,
        guardar_datos_limpios,
        NetflixRegressionPipeline,
        NetflixDecisionTreePipeline
    )

controller_bp = Blueprint("controller_bp", __name__)

# Instancias compartidas en memoria
pipeline_regresion = NetflixRegressionPipeline()
pipeline_arbol = NetflixDecisionTreePipeline()
_datos_inicializados = False

def asegurar_modelos_entrenados():
    global _datos_inicializados
    if not _datos_inicializados:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_limpio = os.path.join(base_dir, "netflix_limpio.csv")

        import pandas as pd
        if os.path.exists(ruta_limpio):
            df = pd.read_csv(ruta_limpio)
        else:
            df_crudo = cargar_dataset()
            df, rep = limpiar_datos(df_crudo)
            guardar_datos_limpios(df, reporte=rep)

        pipeline_regresion.entrenar(df)
        pipeline_arbol.entrenar(df)

        # Generar imágenes si no existen
        ruta_img_reg = os.path.join(base_dir, "static", "regresion_lineal_netflix.png")
        ruta_img_arb_mat = os.path.join(base_dir, "static", "matriz_confusion_netflix.png")
        ruta_img_arb_2d = os.path.join(base_dir, "static", "arbol_decision_2d_netflix.png")
        ruta_img_arb_dia = os.path.join(base_dir, "static", "arbol_diagrama_netflix.png")

        if not os.path.exists(ruta_img_reg):
            pipeline_regresion.generar_grafico_recta(ruta_img_reg)
        if not os.path.exists(ruta_img_arb_mat):
            pipeline_arbol.generar_matriz_confusion(ruta_img_arb_mat)
        if not os.path.exists(ruta_img_arb_2d):
            pipeline_arbol.generar_grafico_arbol_2d(ruta_img_arb_2d)
        if not os.path.exists(ruta_img_arb_dia):
            pipeline_arbol.generar_diagrama_arbol(ruta_img_arb_dia)

        _datos_inicializados = True

# ==========================================
# RUTAS DE VISTA (RENDERING)
# ==========================================

@controller_bp.route("/")
def ruta_selector():
    """Pantalla principal de selección de modelo (Hub)."""
    asegurar_modelos_entrenados()
    return render_template(
        "selector.html",
        metricas_reg=pipeline_regresion.metricas,
        metricas_arb=pipeline_arbol.metricas
    )

@controller_bp.route("/regresion")
def ruta_regresion():
    """Vista exclusiva del modelo de Regresión Lineal."""
    asegurar_modelos_entrenados()
    return render_template(
        "regresion.html",
        metricas=pipeline_regresion.metricas
    )

@controller_bp.route("/arbol")
def ruta_arbol():
    """Vista exclusiva del modelo de Árbol de Decisiones."""
    asegurar_modelos_entrenados()
    return render_template(
        "arbol.html",
        metricas=pipeline_arbol.metricas
    )

# ==========================================
# ENDPOINTS DE API (LÓGICA DEL CONTROLADOR)
# ==========================================

@controller_bp.route("/api/predict/regresion", methods=["POST"])
def api_predict_regresion():
    asegurar_modelos_entrenados()
    data = request.get_json() or {}
    try:
        res = pipeline_regresion.predecir(
            tipo=data.get("tipo", "Movie"),
            pais=data.get("pais", "United States"),
            genero=data.get("genero", "Dramas"),
            anio=int(data.get("anio", 2021)),
            duracion=int(data.get("duracion", 95))
        )
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@controller_bp.route("/api/predict/arbol", methods=["POST"])
def api_predict_arbol():
    asegurar_modelos_entrenados()
    data = request.get_json() or {}
    try:
        res = pipeline_arbol.predecir(
            tipo=data.get("tipo", "Movie"),
            pais=data.get("pais", "United States"),
            genero=data.get("genero", "Dramas"),
            anio=int(data.get("anio", 2021)),
            duracion=int(data.get("duracion", 95))
        )
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@controller_bp.route("/api/predict/arbol-2d", methods=["POST"])
def api_predict_arbol_2d():
    asegurar_modelos_entrenados()
    data = request.get_json() or {}
    try:
        duracion = float(data.get("duracion", 95))
        res = pipeline_arbol.predecir_2d(duracion)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@controller_bp.route("/api/images/regresion")
@controller_bp.route("/api/regression-image")
def api_imagen_regresion():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "regresion_lineal_netflix.png")
    if not os.path.exists(ruta):
        pipeline_regresion.generar_grafico_recta(ruta)
    return send_file(ruta, mimetype="image/png")

@controller_bp.route("/api/images/arbol-2d")
@controller_bp.route("/api/tree-image")
def api_imagen_arbol_2d():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "arbol_decision_2d_netflix.png")
    if not os.path.exists(ruta):
        pipeline_arbol.generar_grafico_arbol_2d(ruta)
    return send_file(ruta, mimetype="image/png")

@controller_bp.route("/api/images/arbol-diagrama")
def api_imagen_arbol_diagrama():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "arbol_diagrama_netflix.png")
    if not os.path.exists(ruta):
        pipeline_arbol.generar_diagrama_arbol(ruta)
    return send_file(ruta, mimetype="image/png")

@controller_bp.route("/api/images/arbol")
@controller_bp.route("/api/images/arbol-matriz")
def api_imagen_arbol():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "matriz_confusion_netflix.png")
    if not os.path.exists(ruta):
        pipeline_arbol.generar_matriz_confusion(ruta)
    return send_file(ruta, mimetype="image/png")
