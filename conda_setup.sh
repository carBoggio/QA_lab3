#!/bin/bash

# Script para configurar el entorno face_recognition automáticamente
# Uso: bash setup_face_recognition.sh

set -e  # Salir si algún comando falla

echo "🔍 Verificando instalación de Conda..."

# Verificar si conda está instalado
if ! command -v conda &> /dev/null; then
    echo "❌ Conda no está instalado"
    echo "📥 Descargando e instalando Miniconda..."
    
    # Detectar arquitectura del sistema
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
    else
        echo "❌ Error: Arquitectura no soportada: $ARCH"
        exit 1
    fi
    
    # Descargar Miniconda
    echo "📥 Descargando Miniconda para $ARCH..."
    wget -O /tmp/miniconda.sh "$MINICONDA_URL"
    
    # Instalar Miniconda
    echo "⚙️  Instalando Miniconda..."
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    
    # Limpiar archivo temporal
    rm /tmp/miniconda.sh
    
    # Inicializar conda
    echo "🔧 Inicializando Conda..."
    "$HOME/miniconda3/bin/conda" init bash
    
    # Recargar el shell
    source "$HOME/.bashrc"
    
    echo "✅ Miniconda instalado exitosamente"
    echo "💡 Nota: Es posible que necesites reiniciar tu terminal o ejecutar 'source ~/.bashrc'"
else
    echo "✅ Conda ya está instalado"
fi

# Asegurar que conda esté disponible en el PATH actual
if ! command -v conda &> /dev/null; then
    # Intentar cargar conda desde ubicaciones comunes
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "❌ Error: No se puede encontrar conda después de la instalación"
        echo "💡 Intenta ejecutar: source ~/.bashrc"
        exit 1
    fi
fi

echo "🔍 Verificando entorno face_recognition..."

# Verificar si el entorno existe
if conda env list | grep -q "^face_recognition "; then
    echo "✅ Entorno face_recognition ya existe"
else
    echo "❌ Entorno face_recognition no existe"
    
    # Verificar si existe environment.yml
    if [ ! -f "environment.yml" ]; then
        echo "❌ Error: environment.yml no encontrado en el directorio actual"
        echo "💡 Asegúrate de estar en el directorio del proyecto con environment.yml"
        exit 1
    fi
    
    echo "🔄 Creando entorno face_recognition..."
    conda env create -f environment.yml
    
    if [ $? -eq 0 ]; then
        echo "✅ Entorno face_recognition creado exitosamente"
    else
        echo "❌ Error al crear el entorno"
        exit 1
    fi
fi

echo "🚀 Activando entorno face_recognition..."

# Activar el entorno conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate face_recognition

# Verificar que el entorno está activo
if [ "$CONDA_DEFAULT_ENV" = "face_recognition" ]; then
    echo "✅ Entorno face_recognition activado correctamente"
else
    echo "❌ Error al activar el entorno"
    exit 1
fi

echo "⚙️  Configurando variables de entorno para TensorFlow..."

# Configurar variables de entorno críticas para CUDA/cuDNN desde conda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="$CONDA_PREFIX"

# Configurar variables de entorno para TensorFlow GPU
export TF_GPU_MEMORY_LIMIT=6000
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_GPU_GARBAGE_COLLECTION=true

# Silenciar warnings molestos (opcional)
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1

echo "✅ Variables de entorno configuradas:"
echo "   - CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "   - CUDA_HOME: $CUDA_HOME"
echo "   - TF_GPU_MEMORY_LIMIT: $TF_GPU_MEMORY_LIMIT"
echo "   - TF_FORCE_GPU_ALLOW_GROWTH: $TF_FORCE_GPU_ALLOW_GROWTH"

echo ""
echo "🎉 ¡Setup completo! Ahora puedes ejecutar:"
echo "   python main.py"
echo ""
echo "💡 Para verificar GPU:"
echo "   python -c \"import tensorflow as tf; print('GPU devices:', len(tf.config.list_physical_devices('GPU')))\""

# Test rápido opcional
echo ""
read -p "¿Quieres hacer un test rápido de GPU? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Testing GPU..."
    python -c "
import tensorflow as tf
print('🔍 TensorFlow version:', tf.__version__)
gpu_devices = tf.config.list_physical_devices('GPU')
print('🎯 GPU devices found:', len(gpu_devices))
if gpu_devices:
    print('✅ GPU disponible!')
    for i, gpu in enumerate(gpu_devices):
        print(f'   GPU {i}: {gpu.name}')
    # Test básico
    try:
        with tf.device('/GPU:0'):
            x = tf.random.normal([100, 100])
            result = tf.matmul(x, x)
        print('✅ GPU test: PASSED')
    except Exception as e:
        print('❌ GPU test: FAILED -', str(e)[:100])
else:
    print('⚠️  No se detectaron GPUs - ejecutándose en CPU')
"
fi

echo ""
echo "🚀 Entorno listo para usar!"