# -*- coding: utf-8 -*-
"""
Punto de Entrada Serverless para Despliegue en Vercel.
Importa y expone la aplicación Flask 'app' para que Vercel gestione las peticiones.
"""

import os
import sys

# Agregar la raíz del repositorio y la carpeta netflix al sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
netflix_dir = os.path.join(root_dir, "netflix")

for path in [root_dir, netflix_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from netflix.app import app

# Vercel espera la variable 'app'
if __name__ == "__main__":
    app.run(debug=True)
