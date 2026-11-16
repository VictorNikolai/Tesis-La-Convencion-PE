@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================================
echo   IRIS 128x128 - Etiquetado de deslizamientos en La Convencion (Cusco)
echo ============================================================================
echo   Servidor web: http://127.0.0.1:5000/segmentation/
echo   Usuario por defecto: admin
echo   Clave por defecto:   laconvencion
echo ============================================================================
start "" "http://127.0.0.1:5000/segmentation/"
python herramientas\iris_128.py abrir --iris iris_128 --usuario admin --clave laconvencion
pause

