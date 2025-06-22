# 🎓 Sistema de Asistencia Automática - Seguridad Informática

Sistema de reconocimiento facial para toma de asistencia automática en la clase de Seguridad Informática.

## 🚀 Características

- ✅ **Reconocimiento facial automático** usando RetinaFace + ArcFace
- ✅ **Normalización facial** con Dlib para mejor precisión
- ✅ **Clasificación KNN** para identificación de estudiantes
- ✅ **Toma de asistencia en tiempo real** con cámara web
- ✅ **Procesamiento de imágenes** para asistencia offline
- ✅ **Reportes automáticos** en formato JSON
- ✅ **Interfaz visual** con información en tiempo real

## 📋 Requisitos

- Python 3.8+
- Cámara web (para modo en vivo)
- GPU recomendada para mejor rendimiento
- Entorno conda configurado (`face_recognition`)

## 🎯 Modos de Uso

### 1. 📹 Asistencia en Tiempo Real (Recomendado)

Ejecuta el sistema con la cámara web para tomar asistencia en vivo:

```bash
# Activar entorno
conda activate face_recognition

# Ejecutar asistencia en vivo
python live_attendance.py
```

**Controles:**
- `q` - Salir y generar reporte final
- `s` - Guardar captura actual
- `r` - Mostrar resumen actual

### 2. 📸 Asistencia desde Imágenes

Procesa imágenes previamente capturadas:

```bash
# Activar entorno
conda activate face_recognition

# Ejecutar asistencia desde imágenes
python attendance_session.py
```

**Configuración:**
Edita `attendance_session.py` y agrega las rutas de tus imágenes:

```python
image_paths = [
    "media/lab2/test.jpg",
    "media/lab2/otra_imagen.jpg",
    # Agregar más imágenes aquí
]
```

## 📊 Reportes Generados

### Formato JSON
Los reportes incluyen:

```json
{
  "class_name": "Seguridad Informática",
  "session_date": "2025-06-21",
  "session_start": "14:30:00",
  "session_end": "15:30:00",
  "total_students": 9,
  "present_students": 7,
  "possible_present_students": 1,
  "absent_students": 1,
  "attendance_percentage": 77.78,
  "students": {
    "franco": {
      "first_seen": "14:30:15",
      "last_seen": "15:25:30",
      "presence_status": "presente",
      "detection_count": 45
    }
  }
}
```

### Información por Estudiante
- **Estado de presencia**: `presente`, `posible_presente`, `ausente`
- **Primera detección**: Hora de primera identificación
- **Última detección**: Hora de última identificación
- **Conteo de detecciones**: Número de veces detectado

## 🎓 Configuración para Seguridad Informática

### 1. Preparar Base de Datos de Estudiantes

Asegúrate de que las imágenes de los estudiantes estén en `media/lab2/`:

```
media/lab2/
├── franco/
│   ├── franco1.jpg
│   ├── franco2.jpg
│   └── ...
├── sergio/
│   ├── sergio1.jpg
│   ├── sergio2.jpg
│   └── ...
└── ...
```

### 2. Entrenar el Clasificador

El sistema entrena automáticamente al iniciar, pero puedes forzar el re-entrenamiento:

```python
from service.recognition_pipeline import FaceRecognitionPipeline

pipeline = FaceRecognitionPipeline()
pipeline.train_classifier()
```

### 3. Configurar Umbrales de Confianza

Edita las variables de entorno para ajustar la sensibilidad:

```bash
export TRESHOLD_PRESENTE=0.6        # Umbral para "presente"
export TRESHOLD_POSIBLE_PRESENTE=0.3 # Umbral para "posible presente"
```

## 🔧 Solución de Problemas

### Error de Memoria GPU
```bash
# Reducir uso de memoria
export TF_GPU_MEMORY_LIMIT=2048
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### Cámara no detectada
```bash
# Verificar dispositivos de video
ls /dev/video*

# Cambiar ID de cámara en el código
session.run_live_session(camera_id=1)  # Probar diferentes IDs
```

### Baja precisión de reconocimiento
1. **Mejorar calidad de imágenes de entrenamiento**
2. **Aumentar número de fotos por estudiante**
3. **Asegurar buena iluminación**
4. **Verificar que las caras estén bien recortadas**

## 📈 Estadísticas de Rendimiento

### Precisión Esperada
- **Condiciones óptimas**: 95%+ de precisión
- **Condiciones normales**: 85-90% de precisión
- **Condiciones difíciles**: 70-80% de precisión

### Factores que Afectan la Precisión
- ✅ **Iluminación adecuada**
- ✅ **Cara frontal y clara**
- ✅ **Distancia apropiada (1-3 metros)**
- ✅ **Calidad de imágenes de entrenamiento**
- ❌ **Iluminación muy baja o muy alta**
- ❌ **Cara de perfil o parcialmente oculta**
- ❌ **Distancia muy lejana o muy cercana**

## 🎯 Casos de Uso para Seguridad Informática

### 1. Asistencia Regular
- Toma de asistencia al inicio de clase
- Monitoreo durante la sesión
- Registro de llegadas tardías

### 2. Laboratorios Prácticos
- Verificación de presencia en laboratorios
- Control de acceso a equipos
- Seguimiento de participación

### 3. Evaluaciones
- Verificación de identidad en exámenes
- Control de acceso a salas de evaluación
- Prevención de suplantación

### 4. Investigación
- Análisis de patrones de asistencia
- Estadísticas de participación
- Reportes para administración

## 📞 Soporte

Para problemas técnicos o consultas sobre el sistema:

1. **Verificar logs**: Los errores se registran automáticamente
2. **Revisar configuración**: Asegurar que el entorno esté correctamente configurado
3. **Probar con imágenes simples**: Usar fotos individuales para debugging

## 🔒 Privacidad y Seguridad

- ✅ **Datos locales**: Toda la información se procesa localmente
- ✅ **Sin transmisión**: No se envían datos a servidores externos
- ✅ **Control total**: Acceso completo a los datos generados
- ✅ **Cumplimiento**: Respeta regulaciones de privacidad educativa

---

**🎓 Sistema desarrollado para la clase de Seguridad Informática**
**📅 Última actualización**: Junio 2025 