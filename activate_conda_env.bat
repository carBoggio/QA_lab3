@echo off
REM Script para configurar el entorno face_recognition automaticamente en Windows
REM Uso: setup_face_recognition.bat

setlocal enabledelayedexpansion

echo 🔍 Verificando instalacion de Conda...

REM Verificar si conda esta instalado
where conda >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Conda no esta instalado
    echo 📥 Descargando e instalando Miniconda...
    
    REM Detectar arquitectura del sistema
    if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
        set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        set "ARCH=x86_64"
    ) else if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
        set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-arm64.exe"
        set "ARCH=ARM64"
    ) else (
        echo ❌ Error: Arquitectura no soportada: %PROCESSOR_ARCHITECTURE%
        pause
        exit /b 1
    )
    
    REM Descargar Miniconda
    echo 📥 Descargando Miniconda para !ARCH!...
    powershell -Command "Invoke-WebRequest -Uri '!MINICONDA_URL!' -OutFile '%TEMP%\miniconda.exe'"
    
    if !errorlevel! neq 0 (
        echo ❌ Error al descargar Miniconda
        pause
        exit /b 1
    )
    
    REM Instalar Miniconda silenciosamente
    echo ⚙️  Instalando Miniconda...
    "%TEMP%\miniconda.exe" /InstallationType=JustMe /RegisterPython=0 /S /D=%USERPROFILE%\miniconda3
    
    if !errorlevel! neq 0 (
        echo ❌ Error al instalar Miniconda
        pause
        exit /b 1
    )
    
    REM Limpiar archivo temporal
    del "%TEMP%\miniconda.exe"
    
    REM Agregar conda al PATH para esta sesion
    set "PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%PATH%"
    
    REM Inicializar conda
    echo 🔧 Inicializando Conda...
    call "%USERPROFILE%\miniconda3\Scripts\conda.exe" init cmd.exe
    
    echo ✅ Miniconda instalado exitosamente
    echo 💡 Nota: Es posible que necesites reiniciar tu terminal
) else (
    echo ✅ Conda ya esta instalado
)

REM Asegurar que conda este disponible
where conda >nul 2>&1
if !errorlevel! neq 0 (
    REM Intentar cargar conda desde ubicaciones comunes
    if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
        set "PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%PATH%"
    ) else if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
        set "PATH=%USERPROFILE%\anaconda3;%USERPROFILE%\anaconda3\Scripts;%PATH%"
    ) else (
        echo ❌ Error: No se puede encontrar conda despues de la instalacion
        echo 💡 Intenta reiniciar tu terminal
        pause
        exit /b 1
    )
)

echo 🔍 Verificando entorno face_recognition...

REM Verificar si el entorno existe
conda env list | findstr "^face_recognition " >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Entorno face_recognition no existe
    
    REM Verificar si existe environment.yml
    if not exist "environment.yml" (
        echo ❌ Error: environment.yml no encontrado en el directorio actual
        echo 💡 Asegurate de estar en el directorio del proyecto con environment.yml
        pause
        exit /b 1
    )
    
    echo 🔄 Creando entorno face_recognition...
    call conda env create -f environment.yml
    
    if !errorlevel! neq 0 (
        echo ❌ Error al crear el entorno
        pause
        exit /b 1
    )
    
    echo ✅ Entorno face_recognition creado exitosamente
) else (
    echo ✅ Entorno face_recognition ya existe
)

echo 🚀 Activando entorno face_recognition...

REM Activar el entorno conda
call conda activate face_recognition

if !errorlevel! neq 0 (
    echo ❌ Error al activar el entorno
    pause
    exit /b 1
)

REM Verificar que el entorno esta activo
if "%CONDA_DEFAULT_ENV%"=="face_recognition" (
    echo ✅ Entorno face_recognition activado correctamente
) else (
    echo ❌ Error: El entorno no se activo correctamente
    pause
    exit /b 1
)

echo ⚙️  Configurando variables de entorno para TensorFlow...

REM Configurar variables de entorno criticas para CUDA/cuDNN desde conda
set "LD_LIBRARY_PATH=%CONDA_PREFIX%\lib;%LD_LIBRARY_PATH%"
set "CUDA_HOME=%CONDA_PREFIX%"

REM Configurar variables de entorno para TensorFlow GPU
set "TF_GPU_MEMORY_LIMIT=6000"
set "TF_FORCE_GPU_ALLOW_GROWTH=true"
set "TF_ENABLE_GPU_GARBAGE_COLLECTION=true"

REM Silenciar warnings molestos (opcional)
set "TF_ENABLE_ONEDNN_OPTS=0"
set "TF_CPP_MIN_LOG_LEVEL=1"

echo ✅ Variables de entorno configuradas:
echo    - CONDA_DEFAULT_ENV: %CONDA_DEFAULT_ENV%
echo    - CUDA_HOME: %CUDA_HOME%
echo    - TF_GPU_MEMORY_LIMIT: %TF_GPU_MEMORY_LIMIT%
echo    - TF_FORCE_GPU_ALLOW_GROWTH: %TF_FORCE_GPU_ALLOW_GROWTH%

echo.
echo 🎉 ¡Setup completo! Ahora puedes ejecutar:
echo    python main.py
echo.
echo 💡 Para verificar GPU:
echo    python -c "import tensorflow as tf; print('GPU devices:', len(tf.config.list_physical_devices('GPU')))"

REM Test rapido opcional
echo.
set /p "reply=¿Quieres hacer un test rapido de GPU? (y/n): "
if /i "!reply!"=="y" (
    echo 🧪 Testing GPU...
    python -c "import tensorflow as tf; print('🔍 TensorFlow version:', tf.__version__); gpu_devices = tf.config.list_physical_devices('GPU'); print('🎯 GPU devices found:', len(gpu_devices)); [print('✅ GPU disponible!') if gpu_devices else print('⚠️  No se detectaron GPUs - ejecutandose en CPU')]; [print(f'   GPU {i}: {gpu.name}') for i, gpu in enumerate(gpu_devices)] if gpu_devices else None; exec('try:\n    with tf.device(\"/GPU:0\"):\n        x = tf.random.normal([100, 100])\n        result = tf.matmul(x, x)\n    print(\"✅ GPU test: PASSED\")\nexcept Exception as e:\n    print(\"❌ GPU test: FAILED -\", str(e)[:100])') if gpu_devices else None"
)

echo.
echo 🚀 Entorno listo para usar!
echo.
echo 💡 IMPORTANTE: Este terminal ahora tiene el entorno activado.
echo    Para usar en futuros terminales, ejecuta: conda activate face_recognition
pause