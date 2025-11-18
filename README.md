# Sistema de Reconocimiento de Gestos BLE

Sistema de reconocimiento de gestos que utiliza sensores IMU conectados por Bluetooth Low Energy (BLE). Desarrollado en Rust para el procesamiento de datos, con integración de modelos de Machine Learning en Python para la clasificación y traducción de gestos.

## Estructura del proyecto

```
├── src/                       # Código fuente Rust
│   ├── main.rs                # Aplicación principal
│   ├── ble.rs                 # Módulo de comunicación BLE
│   ├── gesture_buffer.rs      # Gestión de buffers de datos
│   ├── gesture_extractor.rs   # Extracción de características
│   └── hid.rs                 # Interfaz de salida HID
├── python/                    # Pipeline de clasificación
│   ├── gesture_infer.py       # Script de inferencia
│   ├── best_pipeline__*.joblib # Modelos entrenados
│   └── classes.json           # Etiquetas de gestos
├── gestos/                    # Datos de entrenamiento organizados por gesto
│   ├── gesto-drop/
│   ├── gesto-grab/
│   ├── gesto-slide-derecha/
│   ├── gesto-slide-izquierda/
│   ├── gesto-zoom-in/
│   └── gesto-zoom-out/
├── Cargo.toml                 # Configuración del proyecto Rust
└── README.md
```

## Flujo de ejecución

1. **Captura BLE**: Conexión a sensores IMU y recepción de datos
2. **Procesamiento**: Acumulación de datos en ventanas de tiempo y extracción de características
3. **Clasificación**: Inferencia de gestos mediante modelos de ML (integración PyO3)
4. **Salida**: Generación de eventos de entrada del sistema o visualización de resultados

## Requisitos del sistema

- **Rust**: Versión estable reciente ([Instalar](https://rustup.rs/))
- **Python**: Versión 3.8 o superior
- **Bluetooth**: Adaptador BLE compatible

## Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd rust
```

### 2. Instalar dependencias de Python
```bash
pip install numpy scipy scikit-learn joblib pandas
```

### 3. Compilar el proyecto
```bash
# Modo desarrollo
cargo build

# Modo producción (optimizado)
cargo build --release
```

## Uso

### Reconocimiento en tiempo continuo
```bash
# Ejecutar con dirección MAC del dispositivo BLE
./target/release/onnx-predictor <MAC_ADDRESS>

# Ejemplo
./target/release/onnx-predictor 28:CD:C1:08:37:69
```

### Inferencia offline con archivos CSV
```bash
python3 python/gesture_infer.py --artifacts python --csv gestos/gesto-drop/000001_plot.csv
```

## Configuración

Los parámetros principales del sistema pueden ajustarse según las necesidades:

- **Umbral de confianza**: Nivel mínimo de certeza para aceptar predicciones
- **Tamaño de ventana**: Cantidad de muestras para análisis temporal
- **Umbral de movimiento**: Sensibilidad de detección de gestos
- **Sensores activos**: Configuración de dispositivos IMU

Consulta los archivos de código fuente para modificar estos parámetros.

## Características principales

- **Procesamiento continuo**: Captura y análisis continuo de datos IMU
- **Integración Rust-Python**: Combinación de rendimiento de Rust con ecosistema ML de Python
- **Múltiples sensores**: Soporte para configuraciones multi-sensor
- **Estabilización de predicciones**: Sistema de votación para reducir falsos positivos
- **Interfaz HID**: Generación de eventos de entrada del sistema
- **Formato flexible**: Capacidad de procesar datos desde dispositivos BLE o archivos CSV

## Gestos soportados

El sistema puede reconocer diferentes tipos de gestos, incluyendo:
- Movimientos de agarrar y soltar
- Deslizamientos direccionales
- Gestos de zoom

Los gestos específicos y sus configuraciones se encuentran en el directorio `gestos/`.

## Arquitectura técnica

### Comunicación BLE
- Protocolo de bajo nivel para comunicación con sensores IMU
- Manejo de múltiples dispositivos simultáneos
- Procesamiento asíncrono de frames

### Pipeline de clasificación
- Extracción de características temporales y espectrales
- Modelos de Machine Learning optimizados
- Procesamiento vectorizado de datos

### Interfaz de salida
- Generación de eventos del sistema
- Compatibilidad con protocolos HID
- Logging y visualización de predicciones

## Resolución de problemas

### Problemas comunes

**Error de módulo Python**
- Verifica que los archivos del pipeline estén en el directorio correcto
- Asegúrate de que todas las dependencias de Python estén instaladas

**Problemas de conexión BLE**
- Confirma que el adaptador Bluetooth esté activo
- Verifica que el dispositivo esté encendido y dentro del rango
- Comprueba los permisos de acceso a Bluetooth en tu sistema

**Rendimiento lento**
- Utiliza la versión `--release` para mejor performance
- Verifica que no haya otros procesos consumiendo recursos
- Considera ajustar los parámetros de ventana y umbrales

**Errores de compilación**
- Actualiza Rust a la versión más reciente
- Verifica que todas las dependencias estén disponibles
- Revisa la compatibilidad de las versiones de las librerías

## Desarrollo

### Estructura de módulos

El proyecto está organizado en módulos independientes que pueden ser modificados según necesidades:

- **BLE**: Gestión de comunicación con sensores
- **Buffer**: Manejo de datos temporales
- **Extractor**: Procesamiento de características
- **HID**: Interfaz con el sistema operativo
- **Main**: Orquestación del flujo general

### Añadir nuevos gestos

1. Captura datos del gesto en formato CSV
2. Organiza los archivos en el directorio `gestos/`
3. Actualiza el modelo de clasificación según sea necesario
4. Ajusta los parámetros de reconocimiento si es requerido

## Licencia

Consulta el archivo LICENSE.txt para más información.

## Notas adicionales

- El sistema utiliza PyO3 para integración directa Rust-Python sin overhead de IPC
- Los datos IMU incluyen acelerómetro y cuaterniones de orientación
- El procesamiento puede funcionar con sensores faltantes mediante manejo de opcionales
- La arquitectura permite agregar nuevos tipos de salida o procesamiento fácilmente
