# -*- coding: utf-8 -*-
"""
Paquete Models - Capa de Datos y Modelos ML (MVC)
"""

from .limpieza_model import cargar_dataset, limpiar_datos, guardar_datos_limpios
from .regresion_model import NetflixRegressionPipeline
from .arbol_model import NetflixDecisionTreePipeline
from .svm_model import NetflixSVMPipeline
from .rna_model import NetflixRNAPipeline
