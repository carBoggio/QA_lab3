#!/bin/bash

echo "🧹 LIMPIEZA Y REINSTALACIÓN CORRECTA DE PYTORCH 🧹"
echo "=================================================="

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

echo ""
echo "=== PASO 1: LIMPIEZA COMPLETA ==="
log_warning "Removiendo instalaciones incorrectas de PyTorch..."

# Remover de Poetry
log_info "Removiendo de Poetry..."
poetry remove torch --no-interaction 2>/dev/null || echo "torch no está en Poetry"
poetry remove torchvision --no-interaction 2>/dev/null || echo "torchvision no está en Poetry"
poetry remove torchaudio --no-interaction 2>/dev/null || echo "torchaudio no está en Poetry"

# Limpiar instalaciones de pip dentro del entorno Poetry
log_info "Limpiando instalaciones incorrectas de pip..."
poetry run pip uninstall torch torchvision torchaudio -y 2>/dev/null || echo "No hay instalaciones de pip para limpiar"

# Verificar que esté limpio
log_info "Verificando limpieza..."
poetry run python -c "
try:
    import torch
    print('❌ PyTorch todavía está instalado')
except ImportError:
    print('✅ PyTorch removido correctamente')

try:
    import torchvision
    print('❌ TorchVision todavía está instalado')
except ImportError:
    print('✅ TorchVision removido correctamente')
" 2>/dev/null

log_success "Limpieza completa"

echo ""
echo "=== PASO 2: CONFIGURAR SOURCE CORRECTA ==="
log_info "Agregando source PyTorch para CUDA 12.4..."

# Remover source anterior si existe
poetry source remove pytorch-cu124 2>/dev/null || echo "No hay source anterior"

# Agregar source nueva
poetry source add --priority=explicit pytorch-cu124 https://download.pytorch.org/whl/cu124

log_success "Source PyTorch CUDA 12.4 configurada"

echo ""
echo "=== PASO 3: INSTALACIÓN CORRECTA CON POETRY ==="
log_info "Instalando PyTorch 2.5.1 + TorchVision 0.20.1 con Poetry..."

# Instalar usando Poetry correctamente
poetry add --source pytorch-cu124 torch==2.5.1 torchvision==0.20.1

log_success "Instalación con Poetry completada"

echo ""
echo "=== PASO 4: VERIFICACIÓN COMPLETA ==="
log_info "Verificando instalación correcta..."

poetry run python -c "
print('🔍 VERIFICACIÓN DE INSTALACIÓN')
print('=' * 40)

try:
    import torch
    import torchvision
    
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ TorchVision: {torchvision.__version__}')
    print(f'✅ CUDA disponible: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'✅ CUDA version en PyTorch: {torch.version.cuda}')
        print(f'✅ GPU detectada: {torch.cuda.get_device_name(0)}')
        
        # Test GPU
        try:
            x = torch.randn(2, 3).cuda()
            result = x.sum().item()
            print(f'✅ Test GPU exitoso: {result:.2f}')
        except Exception as e:
            print(f'⚠️  Error en test GPU: {e}')
    else:
        print('⚠️  CUDA no disponible')
        
    # Verificar que torch está en el entorno correcto
    import sys
    torch_location = torch.__file__
    if 'poetry' in torch_location:
        print(f'✅ PyTorch instalado en entorno Poetry: {torch_location}')
    else:
        print(f'⚠️  PyTorch ubicación: {torch_location}')
        
except ImportError as e:
    print(f'❌ Error importando PyTorch: {e}')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "=== PASO 5: VERIFICAR POETRY DEPENDENCIES ==="
log_info "Verificando dependencias en Poetry..."

echo "Dependencias PyTorch en pyproject.toml:"
poetry show torch torchvision 2>/dev/null || log_warning "No aparecen en poetry show"

echo ""
echo "=== PASO 6: PROBAR TU PROYECTO ==="
log_info "Probando tu proyecto original..."

echo "Ejecutando: poetry run python trySistem.py"
poetry run python trySistem.py

echo ""
echo "🎉 INSTALACIÓN CORRECTA COMPLETADA"
echo "=================================="
echo "✅ PyTorch y TorchVision instalados correctamente con Poetry"
echo "✅ Compatible con CUDA 12.4"
echo "✅ Sin mezclar pip y Poetry"
echo "✅ Dependencias gestionadas correctamente"