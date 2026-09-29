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
        NetflixDecisionTreePipeline,
        NetflixSVMPipeline,
        NetflixRNAPipeline
    )
except (ImportError, ValueError):
    from ..models import (
        cargar_dataset,
        limpiar_datos,
        guardar_datos_limpios,
        NetflixRegressionPipeline,
        NetflixDecisionTreePipeline,
        NetflixSVMPipeline,
        NetflixRNAPipeline
    )

controller_bp = Blueprint("controller_bp", __name__)

# Instancias compartidas en memoria
pipeline_regresion = NetflixRegressionPipeline()
pipeline_arbol = NetflixDecisionTreePipeline()
pipeline_svm = NetflixSVMPipeline()
pipeline_rna = NetflixRNAPipeline()
_datos_inicializados = False

def asegurar_modelos_entrenados():
    global _datos_inicializados, pipeline_regresion, pipeline_arbol, pipeline_svm, pipeline_rna
    if not _datos_inicializados:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_limpio = os.path.join(base_dir, "netflix_limpio.csv")

        # Intentar cargar desde caché serializado para despliegue serverless inmediato
        import pickle
        cache_candidatos = [
            os.path.join(base_dir, "models_cache.pkl"),
            os.path.join(os.path.dirname(base_dir), "models_cache.pkl"),
            os.path.join(base_dir, "models", "models_cache.pkl"),
            "models_cache.pkl"
        ]
        loaded_from_cache = False
        for cpath in cache_candidatos:
            if os.path.exists(cpath):
                try:
                    with open(cpath, "rb") as f:
                        cdata = pickle.load(f)
                        pipeline_regresion = cdata.get("regresion", pipeline_regresion)
                        pipeline_arbol = cdata.get("arbol", pipeline_arbol)
                        pipeline_svm = cdata.get("svm", pipeline_svm)
                        pipeline_rna = cdata.get("rna", pipeline_rna)
                        if pipeline_regresion.is_fitted and pipeline_arbol.is_fitted and pipeline_svm.is_fitted and getattr(pipeline_rna, "is_fitted", False):
                            loaded_from_cache = True
                            break
                except Exception:
                    pass

        if not loaded_from_cache:
            import pandas as pd
            if os.path.exists(ruta_limpio):
                df = pd.read_csv(ruta_limpio)
            else:
                df_crudo = cargar_dataset()
                df, rep = limpiar_datos(df_crudo)
                guardar_datos_limpios(df, reporte=rep)

            pipeline_regresion.entrenar(df)
            pipeline_arbol.entrenar(df)
            pipeline_svm.entrenar(df)
            pipeline_rna.entrenar(df)

        # Generar imágenes si no existen
        ruta_img_reg = os.path.join(base_dir, "static", "regresion_lineal_netflix.png")
        ruta_img_arb_mat = os.path.join(base_dir, "static", "matriz_confusion_netflix.png")
        ruta_img_arb_2d = os.path.join(base_dir, "static", "arbol_decision_2d_netflix.png")
        ruta_img_arb_dia = os.path.join(base_dir, "static", "arbol_diagrama_netflix.png")
        ruta_img_svm_hip = os.path.join(base_dir, "static", "svm_hiperplano_netflix.png")
        ruta_img_svm_mat = os.path.join(base_dir, "static", "matriz_confusion_svm_netflix.png")
        ruta_img_svm_dur = os.path.join(base_dir, "static", "svm_duracion_edad_netflix.png")
        ruta_img_rna_dur = os.path.join(base_dir, "static", "rna_duracion_edad_netflix.png")
        ruta_img_rna_loss = os.path.join(base_dir, "static", "rna_curva_perdida_netflix.png")
        ruta_img_rna_mat = os.path.join(base_dir, "static", "matriz_confusion_rna_netflix.png")

        if not os.path.exists(ruta_img_reg):
            pipeline_regresion.generar_grafico_recta(ruta_img_reg)
        if not os.path.exists(ruta_img_arb_mat):
            pipeline_arbol.generar_matriz_confusion(ruta_img_arb_mat)
        if not os.path.exists(ruta_img_arb_2d):
            pipeline_arbol.generar_grafico_arbol_2d(ruta_img_arb_2d)
        if not os.path.exists(ruta_img_arb_dia):
            pipeline_arbol.generar_diagrama_arbol(ruta_img_arb_dia)
        if not os.path.exists(ruta_img_svm_dur):
            pipeline_svm.generar_grafico_duracion_edad(ruta_img_svm_dur)
        if not os.path.exists(ruta_img_svm_hip):
            pipeline_svm.generar_grafico_hiperplano(ruta_img_svm_hip)
        if not os.path.exists(ruta_img_svm_mat):
            pipeline_svm.generar_matriz_confusion(ruta_img_svm_mat)
        if not os.path.exists(ruta_img_rna_dur):
            pipeline_rna.generar_grafico_duracion_edad(ruta_img_rna_dur)
        if not os.path.exists(ruta_img_rna_loss):
            pipeline_rna.generar_grafico_perdida(ruta_img_rna_loss)
        if not os.path.exists(ruta_img_rna_mat):
            pipeline_rna.generar_matriz_confusion(ruta_img_rna_mat)

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
        metricas_arb=pipeline_arbol.metricas,
        metricas_svm=pipeline_svm.metricas,
        metricas_rna=pipeline_rna.metricas
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

@controller_bp.route("/svm")
def ruta_svm():
    """Vista exclusiva del modelo de Máquinas de Vectores de Soporte (SVM)."""
    asegurar_modelos_entrenados()
    return render_template(
        "svm.html",
        metricas=pipeline_svm.metricas
    )

@controller_bp.route("/rna")
def ruta_rna():
    """Vista exclusiva del modelo de Redes Neuronales Artificiales (RNA)."""
    asegurar_modelos_entrenados()
    return render_template(
        "rna.html",
        metricas=pipeline_rna.metricas
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

@controller_bp.route("/api/predict/svm", methods=["POST"])
def api_predict_svm():
    asegurar_modelos_entrenados()
    data = request.get_json() or {}
    try:
        res = pipeline_svm.predecir(
            tipo=data.get("tipo", "Movie"),
            pais=data.get("pais", "United States"),
            genero=data.get("genero", "Dramas"),
            anio=int(data.get("anio", 2021)),
            duracion=int(data.get("duracion", 95))
        )
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@controller_bp.route("/api/predict/svm-2d", methods=["POST"])
def api_predict_svm_2d():
    asegurar_modelos_entrenados()
    data = request.get_json() or {}
    try:
        duracion = float(data.get("duracion", 95))
        anio = float(data.get("anio", 2021))
        res = pipeline_svm.predecir_2d(duracion=duracion, anio=anio)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@controller_bp.route("/api/predict/svm-duracion", methods=["POST"])
def api_predict_svm_duracion():
    asegurar_modelos_entrenados()
    data = request.get_json() or {}
    try:
        duracion = float(data.get("duracion", 120))
        res = pipeline_svm.predecir_duracion(duracion)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@controller_bp.route("/api/images/svm-2d")
@controller_bp.route("/api/images/svm-duracion")
def api_imagen_svm_2d():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "svm_duracion_edad_netflix.png")
    if not os.path.exists(ruta):
        pipeline_svm.generar_grafico_duracion_edad(ruta)
    return send_file(ruta, mimetype="image/png")

@controller_bp.route("/api/images/svm-hiperplano")
@controller_bp.route("/api/svm-image")
def api_imagen_svm_hiperplano():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "svm_hiperplano_netflix.png")
    if not os.path.exists(ruta):
        pipeline_svm.generar_grafico_hiperplano(ruta)
    return send_file(ruta, mimetype="image/png")

@controller_bp.route("/api/images/svm-matriz")
def api_imagen_svm_matriz():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "matriz_confusion_svm_netflix.png")
    if not os.path.exists(ruta):
        pipeline_svm.generar_matriz_confusion(ruta)
    return send_file(ruta, mimetype="image/png")

# ==========================================
# RUTAS API: RED NEURONAL ARTIFICIAL (RNA)
# ==========================================

@controller_bp.route("/api/predict/rna-duracion", methods=["POST"])
@controller_bp.route("/api/predict/rna", methods=["POST"])
def api_predecir_rna():
    asegurar_modelos_entrenados()
    data = request.get_json() or {}
    try:
        duracion = float(data.get("duracion", 105))
        res = pipeline_rna.predecir_duracion(duracion)
        return jsonify({"status": "success", "data": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@controller_bp.route("/api/images/rna-2d")
@controller_bp.route("/api/images/rna-duracion")
def api_imagen_rna_2d():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "rna_duracion_edad_netflix.png")
    if not os.path.exists(ruta):
        pipeline_rna.generar_grafico_duracion_edad(ruta)
    return send_file(ruta, mimetype="image/png")

@controller_bp.route("/api/images/rna-loss")
def api_imagen_rna_loss():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "rna_curva_perdida_netflix.png")
    if not os.path.exists(ruta):
        pipeline_rna.generar_grafico_perdida(ruta)
    return send_file(ruta, mimetype="image/png")

@controller_bp.route("/api/images/rna-matriz")
def api_imagen_rna_matriz():
    asegurar_modelos_entrenados()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "static", "matriz_confusion_rna_netflix.png")
    if not os.path.exists(ruta):
        pipeline_rna.generar_matriz_confusion(ruta)
    return send_file(ruta, mimetype="image/png")



