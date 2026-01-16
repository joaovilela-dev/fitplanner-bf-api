@echo off
echo ========================================
echo   FitPlanner Body Fat API
echo   Iniciando servidor...
echo ========================================
echo.

REM Sobe para a raiz do projeto
cd ..

REM Ativa o ambiente virtual
call venv\Scripts\activate

REM Inicia a API (corrigido: main:app ao invés de app.main:app)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause