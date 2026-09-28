# -*- coding: utf-8 -*-
"""
Punto de Entrada Principal - Netflix Web Interface & ML Studio
Permite ejecutar el servidor web directamente desde la raíz del proyecto.
"""

import os
import sys

# Agregar carpeta 'netflix' al path
base_dir = os.path.dirname(os.path.abspath(__file__))
netflix_dir = os.path.join(base_dir, "netflix")
if netflix_dir not in sys.path:
    sys.path.insert(0, netflix_dir)

from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================", flush=True)
    print(f"  NETFLIX DATA & MACHINE LEARNING STUDIO", flush=True)
    print(f"  Servidor activo en: http://127.0.0.1:{port}", flush=True)
    print(f"=======================================================\n", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False)
