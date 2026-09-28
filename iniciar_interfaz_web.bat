@echo off
title Servidor Web Netflix Data & ML Studio
echo ========================================================
echo   INICIANDO NETFLIX DATA STUDIO & ML PIPELINE
echo   Abriendo servidor local en http://127.0.0.1:5000 ...
echo ========================================================
start http://127.0.0.1:5000
python netflix/app.py
pause
