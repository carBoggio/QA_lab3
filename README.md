# Face Recognition Detection Project

Un sistema de reconocimiento facial desarrollado con Python que utiliza Deep Learning para detectar e identificar personas en imágenes.

## 🚀 Configuración del entorno

Este proyecto utiliza **Conda** para la gestión completa de dependencias:
- **Conda**: Maneja dependencias del sistema (CUDA, cuDNN, OpenCV) y Python (TensorFlow, DeepFace, scikit-learn)

### 📋 Requisitos previos

- Python 3.10+
- NVIDIA GPU (opcional, pero recomendado)
- Miniconda o Anaconda

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd face_recogniton_detection_proyect
```

### 2. Crear entorno desde archivo
```bash
# Crear entorno desde environment.yml
conda env create -f environment.yml

# Activar entorno
conda activate face_recognition
```

### 3. Verificar instalación completa
```bash
# Test completo del setup
conda activate face_recognition

# Verificar configuración
python -c "
import os
import tensorflow as tf

print('🔍 VERIFICACIÓN COMPLETA:')
print('=' * 50)

# Variables de entorno
print('✓ Entorno conda:', os.environ.get('CONDA_DEFAULT_ENV'))
print('✓ CUDA_HOME:', os.environ.get('CUDA_HOME', 'Not set'))
print('✓ Memory limit:', os.environ.get('TF_GPU_MEMORY_LIMIT', 'Not set'))

# TensorFlow y GPU
print('✓ TensorFlow:', tf.__version__)
gpu_devices = tf.config.list_physical_devices('GPU')
print(f'✓ GPU devices: {len(gpu_devices)}')

# Test cuDNN
if gpu_devices:
    try:
        with tf.device('/GPU:0'):
            x = tf.random.normal([1, 256, 256, 3])
            conv = tf.keras.layers.Conv2D(32, 3)(x)
        print('✅ cuDNN test: PASSED')
    except Exception as e:
        print('❌ cuDNN test: FAILED')
        print('💡 Revisa configuración de memoria GPU')
else:
    print('ℹ️ Ejecutándose en modo CPU')

print('\\n🎉 Setup verification complete!')
"

# Test de reconocimiento facial (opcional)
python -c "
from PIL import Image
from service.recognition_pipeline import FaceRecognitionPipeline

try:
    pipeline = FaceRecognitionPipeline()
    print('✅ Pipeline cargado correctamente')
    
    # Test con imagen pequeña
    img = Image.new('RGB', (300, 300), color='red')
    faces = pipeline.detect_faces_in_image(img)
    print(f'✅ Detección funcionando (caras encontradas: {len(faces)})')
except Exception as e:
    print(f'⚠️ Error en pipeline: {str(e)[:100]}')
    print('💡 Verifica que todas las dependencias estén instaladas')
"
```

## 📂 Estructura del proyecto

```
face_recogniton_detection_proyect/
├── media/                          # Imágenes y datos
│   ├── lab2/                      # Fotos de entrenamiento por persona
│   │   ├── chico_barba/           # Imágenes persona 1
│   │   ├── franco/                # Imágenes persona 2
│   │   ├── larisa/                # Imágenes persona 3
│   │   └── ...
│   ├── test_images/               # Imágenes de prueba
│   └── test.jpg                   # Imagen de test principal
├── service/                       # Código principal
│   ├── clasifier/                 # Clasificadores
│   ├── detection/                 # Detección facial
│   ├── recognition/               # Reconocimiento
│   ├── recognition_pipeline.py    # Pipeline principal
│   └── utils.py                   # Utilidades
├── main.py                        # Script principal
├── environment.yml                # Dependencias Conda
└── README.md                      # Este archivo
```

## 🎮 Uso diario

### Activar entorno
```bash
# SIEMPRE activar el entorno antes de trabajar
conda activate face_recognition

# Verificar que estás en el entorno correcto
echo $CONDA_DEFAULT_ENV
# Debería mostrar: face_recognition
```

### Ejecutar el proyecto

#### 🚀 Modo normal (recomendado)
```bash
# Activar entorno
conda activate face_recognition

# Configurar variables de entorno (si no están en .bashrc)
export TF_GPU_MEMORY_LIMIT=5120
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Ejecutar reconocimiento facial
python main.py
```

#### 🖼️ Modo imágenes grandes
```bash
# Para imágenes >1000px (mayor uso de memoria)
conda activate face_recognition
export TF_GPU_MEMORY_LIMIT=6000
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_GPU_GARBAGE_COLLECTION=true

python main.py
```

#### 🔧 Modo debugging (solo CPU)
```bash
# Para debugging o hardware limitado
conda activate face_recognition
export CUDA_VISIBLE_DEVICES=""

python main.py
```

#### ⚡ Modo automático (configuración permanente)
```bash
# Si configuraste variables en ~/.bashrc
conda activate face_recognition
python main.py  # Variables se cargan automáticamente
```

### Entrenar el clasificador
```bash
# El sistema carga automáticamente las imágenes de media/lab2/
# y entrena el clasificador KNN
python main.py
```

## 📦 Gestión de dependencias

### Agregar dependencias Python
```bash
# Activar entorno
conda activate face_recognition

# Agregar paquete con conda (prioridad)
conda install nombre_paquete -c conda-forge

# Si no está en conda, usar pip
pip install nombre_paquete

# Ejemplo
conda install matplotlib opencv -c conda-forge
pip install deepface  # si necesitas versión específica
```

### Actualizar dependencias
```bash
# Actualizar todo el entorno
conda update --all

# Actualizar paquete específico
conda update nombre_paquete

# Ver paquetes instalados
conda list
```

### Exportar entorno
```bash
# Exportar entorno completo (equivalente a poetry.lock)
conda env export > environment-lock.yml

# Exportar solo dependencias principales
conda env export --from-history > environment.yml
```

## 🔧 Comandos útiles

### Información del entorno
```bash
# Ver entornos conda disponibles
conda env list

# Ver información del entorno activo
conda info

# Ver configuración conda
conda config --show
```

### Debugging
```bash
# Verificar versiones importantes
python -c "
import tensorflow as tf
import deepface
import sklearn
print(f'Python: {tf.__version__}')
print(f'TensorFlow: {tf.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'GPU: {tf.config.list_physical_devices(\"GPU\")}')
"

# Ver ubicación de Python
which python

# Ver paquetes instalados
conda list | grep -E "(tensorflow|deepface|pillow)"
```

### Reset del entorno
```bash
# Si algo se rompe, recrear entorno
conda deactivate
conda env remove -n face_recognition

# Volver a crear desde environment.yml
conda env create -f environment.yml
conda activate face_recognition
```

## 🧪 Testing

### Test básico de GPU
```bash
python -c "
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    print('✅ GPU disponible')
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.matmul(a, a)
    print('✅ Test GPU exitoso')
else:
    print('⚠️ GPU no disponible, usando CPU')
"
```

### Test de reconocimiento
```bash
# Verificar que detecta caras en imagen de test
python -c "
from PIL import Image
from service.recognition_pipeline import FaceRecognitionPipeline

pipeline = FaceRecognitionPipeline()
img = Image.open('media/test.jpg')
faces = pipeline.detect_faces_in_image(img)
print(f'Caras detectadas: {len(faces)}')
"
```

## ⚠️ Troubleshooting

### Problema: `conda: command not found`
```bash
# Solución: Inicializar conda
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Problema: Error de cuDNN version mismatch
```bash
# Síntomas:
# "Loaded runtime CuDNN library: 8.2.4 but source was compiled with: 8.9.4"

# Solución 1: Verificar que cuDNN está en el entorno correcto
conda activate face_recognition
conda list | grep cudnn

# Si no aparece, reinstalar
conda install cudnn -c conda-forge

# Solución 2: Forzar uso de cuDNN de conda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="$CONDA_PREFIX"
```

### Problema: GPU memory errors / OOM (Out of Memory)
```bash
# Síntomas:
# "cuDNN launch failure : input shape ([1,891,1980,3])"
# "CPU->GPU Memcpy failed"
# "Garbage collection: deallocate free memory regions"

# Solución 1: Configurar límites de memoria
export TF_GPU_MEMORY_LIMIT=5120
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Solución 2: Para imágenes muy grandes
export TF_GPU_MEMORY_LIMIT=6000
export TF_ENABLE_GPU_GARBAGE_COLLECTION=true

# Solución 3: Usar CPU temporalmente
export CUDA_VISIBLE_DEVICES=""

# Solución 4: Verificar tamaño de imágenes
python -c "
from PIL import Image
img = Image.open('media/test.jpg')
print(f'Tamaño imagen: {img.size}')
if max(img.size) > 1000:
    print('⚠️ Imagen muy grande, considera redimensionar')
"
```

### Problema: GPU no detectada
```bash
# Verificar driver NVIDIA
nvidia-smi

# Verificar CUDA
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"

# Verificar detección GPU
python -c "
import tensorflow as tf
print('GPU devices:', tf.config.list_physical_devices('GPU'))
"

# Reinstalar tensorflow-gpu si es necesario
conda uninstall tensorflow-gpu
conda install tensorflow-gpu=2.15.0 -c conda-forge
```

### Problema: Conflictos entre conda y pip
```bash
# Síntomas: Versiones inconsistentes de dependencias

# Solución: Verificar origen de paquetes
conda list

# Reinstalar con conda si es posible
conda uninstall paquete_problemático
conda install paquete_problemático -c conda-forge

# Solo usar pip para paquetes que NO están en conda
```

### Problema: Variables de entorno no persisten
```bash
# Agregar variables permanentes a ~/.bashrc
echo 'export TF_GPU_MEMORY_LIMIT=5120' >> ~/.bashrc
echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> ~/.bashrc
source ~/.bashrc

# O usar configuración automática (ver sección Variables de entorno)
```

### Problema: Warnings molestos
```bash
# Silenciar warnings de TensorFlow
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

# Silenciar warnings específicos
export TF_FORCE_GPU_ALLOW_GROWTH=true  # "Overriding orig_value setting"
```

### Problema: Performance muy lenta
```bash
# Verificar que usa GPU
python -c "
import tensorflow as tf
print('GPU disponible:', len(tf.config.list_physical_devices('GPU')))

# Test velocidad
import time
start = time.time()
with tf.device('/GPU:0'):
    x = tf.random.normal([1000, 1000])
    result = tf.matmul(x, x)
print(f'GPU test time: {time.time() - start:.2f}s')
"

# Si es lento, verificar memoria
nvidia-smi

# Optimizar memoria
export TF_GPU_MEMORY_LIMIT=5120
```

## 📝 Variables de entorno necesarias

### 🔧 Configuración automática (recomendada)

Agregar al final de `~/.bashrc` para configuración permanente:

```bash
# Configuración para el entorno face_recognition
if [ "$CONDA_DEFAULT_ENV" = "face_recognition" ]; then
    # CUDA/cuDNN desde conda (CRÍTICO)
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    export CUDA_HOME="$CONDA_PREFIX"
    
    # Optimización de memoria GPU (RECOMENDADO)
    export TF_GPU_MEMORY_LIMIT=5120                    # 5GB límite GPU
    export TF_FORCE_GPU_ALLOW_GROWTH=true              # Memoria dinámica
    
    # Optimización general (OPCIONAL)
    export TF_ENABLE_ONEDNN_OPTS=0                     # Silenciar warnings
    export TF_ENABLE_GPU_GARBAGE_COLLECTION=true       # Limpiar memoria automático
fi
```

Aplicar cambios:
```bash
source ~/.bashrc
```

### ⚡ Configuración manual por sesión

#### Para uso normal (GPU optimizada):
```bash
# Variables críticas
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="$CONDA_PREFIX"

# Memoria GPU optimizada
export TF_GPU_MEMORY_LIMIT=5120        # 5GB - balance perfecto
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Crecimiento dinámico

# Ejecutar proyecto
python main.py
```

#### Para imágenes muy grandes (memoria máxima):
```bash
# Usar casi toda la memoria GPU disponible
export TF_GPU_MEMORY_LIMIT=6000        # 6GB - máximo recomendado
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_GPU_GARBAGE_COLLECTION=true

python main.py
```

#### Para debugging o hardware limitado (solo CPU):
```bash
# Forzar uso de CPU
export CUDA_VISIBLE_DEVICES=""

python main.py
```

#### Para múltiples GPUs:
```bash
# Seleccionar GPU específica
export CUDA_VISIBLE_DEVICES=0          # Usar solo GPU 0
# o
export CUDA_VISIBLE_DEVICES=0,1        # Usar GPUs 0 y 1
```

### 🎯 Configuración según casos de uso

| Caso de uso | Variables recomendadas |
|-------------|------------------------|
| **Desarrollo normal** | `TF_GPU_MEMORY_LIMIT=5120` + `TF_FORCE_GPU_ALLOW_GROWTH=true` |
| **Imágenes muy grandes** | `TF_GPU_MEMORY_LIMIT=6000` + garbage collection |
| **Testing/Debugging** | `CUDA_VISIBLE_DEVICES=""` (solo CPU) |
| **Producción** | Configuración automática en `.bashrc` |
| **Múltiples usuarios** | `CUDA_VISIBLE_DEVICES=X` para asignar GPU específica |

### 🔍 Verificar configuración

```bash
# Script de verificación completa
python -c "
import os
import tensorflow as tf

print('🔍 CONFIGURACIÓN ACTUAL:')
print('=' * 50)
print('CONDA_PREFIX:', os.environ.get('CONDA_PREFIX', 'Not set'))
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'Not set')[:100])
print('CUDA_HOME:', os.environ.get('CUDA_HOME', 'Not set'))
print('TF_GPU_MEMORY_LIMIT:', os.environ.get('TF_GPU_MEMORY_LIMIT', 'Not set'))
print('TF_FORCE_GPU_ALLOW_GROWTH:', os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not set'))
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'All GPUs'))

print('\\n🧪 TENSORFLOW STATUS:')
print('TensorFlow version:', tf.__version__)
print('GPU devices available:', len(tf.config.list_physical_devices('GPU')))

# Test memoria GPU
if tf.config.list_physical_devices('GPU'):
    try:
        with tf.device('/GPU:0'):
            x = tf.random.normal([1, 512, 512, 3])
            conv = tf.keras.layers.Conv2D(32, 3)(x)
        print('✅ GPU memory test: PASSED')
    except Exception as e:
        print('❌ GPU memory test: FAILED -', str(e)[:100])
else:
    print('ℹ️ Running in CPU mode')
"
```

### 🚨 Troubleshooting variables de entorno

#### Error: "cuDNN version mismatch"
```bash
# Solución: Forzar uso de cuDNN de conda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="$CONDA_PREFIX"
```

#### Error: "GPU memory limit exceeded"
```bash
# Solución: Reducir límite de memoria
export TF_GPU_MEMORY_LIMIT=4096  # Reducir a 4GB
```

#### Error: "CUDA out of memory"
```bash
# Solución temporal: Usar CPU
export CUDA_VISIBLE_DEVICES=""

# Solución permanente: Optimizar código para redimensionar imágenes
```

#### Warnings molestos de TensorFlow
```bash
# Silenciar warnings
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL
```

## 🚀 Desarrollo

### Agregar nuevas personas
1. Crear carpeta en `media/lab2/nombre_persona/`
2. Agregar imágenes de la persona en la carpeta
3. Ejecutar `python main.py` para reentrenar

### Modificar pipeline
- El pipeline principal está en `service/recognition_pipeline.py`
- Implementa la clase abstracta `BaseFaceRecognitionPipeline`
- Métodos principales: `detect_faces_in_image`, `extract_embedding_from_single_face_image`, `classify_person_identity_from_embedding`

## 🤝 Contribuir

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Reconocimientos

- [DeepFace](https://github.com/serengil/deepface) - Framework de reconocimiento facial
- [TensorFlow](https://tensorflow.org/) - Machine Learning framework
- [RetinaFace](https://github.com/StanislasBertrand/RetinaFace-tf2) - Detección facial