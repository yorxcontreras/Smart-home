@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python nao foi encontrado. Instale o Python 3.10 ou superior e tente novamente.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Criando o ambiente virtual...
    python -m venv .venv
    if errorlevel 1 goto :error
)

echo Instalando ou atualizando as dependencias...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo Iniciando o Imovel Inteligente...
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b 0

:error
echo Nao foi possivel preparar a aplicacao. Consulte o README.md.
pause
exit /b 1

