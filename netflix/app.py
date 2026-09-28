# -*- coding: utf-8 -*-
"""
Servidor Web Flask - Netflix Machine Learning Studio (Arquitectura MVC)
Orquesta la aplicación registrando los controladores y sirviendo las vistas.
"""

import os
import sys
from flask import Flask

# Asegurar importación del paquete netflix
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from controllers import controller_bp

app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, "templates"),
    static_folder=os.path.join(base_dir, "static")
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Registrar Blueprint del Controlador (MVC)
app.register_blueprint(controller_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================", flush=True)
    print(f"  NETFLIX ML STUDIO (ARQUITECTURA MVC)", flush=True)
    print(f"  Servidor activo en: http://127.0.0.1:{port}", flush=True)
    print(f"  - Selector de Modelos: http://127.0.0.1:{port}/", flush=True)
    print(f"  - Regresión Lineal:   http://127.0.0.1:{port}/regresion", flush=True)
    print(f"  - Árbol de Decisiones: http://127.0.0.1:{port}/arbol", flush=True)
    print(f"=======================================================\n", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False)
