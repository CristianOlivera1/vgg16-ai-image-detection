# Detección de imágenes generadas por IA mediante VGG16

## Carátula

| Campo | Información |
|---|---|
| Institución | `[Nombre de la institución]` |
| Facultad / programa | `[Facultad o programa académico]` |
| Asignatura | `[Nombre de la asignatura]` |
| Título | Detección de imágenes generadas por IA mediante VGG16 |
| Autor(es) | `[Nombre(s) del/de los estudiante(s)]` |
| Docente | `[Nombre del/de la docente]` |
| Ciudad y fecha | `[Ciudad, mes y año]` |

## Índice

1. [Introducción](#1-introducción)
2. [Objetivo](#2-objetivo)
   1. [General](#21-general)
   2. [Específicos](#22-específicos)
3. [Fundamentos teóricos](#3-fundamentos-teóricos)
4. [Comparación con otras arquitecturas](#4-comparación-con-otras-arquitecturas)
5. [Caso práctico](#5-caso-práctico)
   1. [Dataset](#51-dataset)
   2. [Herramientas utilizadas](#52-herramientas-utilizadas)
   3. [Preparación de los datos](#53-preparación-de-los-datos)
   4. [Implementación del modelo](#54-implementación-del-modelo)
   5. [Entrenamiento](#55-entrenamiento)
   6. [Resultados](#56-resultados)
   7. [Análisis de resultados](#57-análisis-de-resultados)
6. [Ventajas y limitaciones](#6-ventajas-y-limitaciones)
7. [Conclusiones y recomendaciones](#7-conclusiones-y-recomendaciones)
8. [Bibliografía](#8-bibliografía)

## 1. Introducción

La generación de imágenes mediante modelos de inteligencia artificial ha
avanzado con rapidez, haciendo más difícil distinguir visualmente imágenes
sintéticas de fotografías reales. La clasificación automática puede apoyar
esta tarea, aunque sus resultados dependen del conjunto de datos, del
preprocesamiento y de la capacidad del modelo para generalizar a fuentes no
vistas.

Este trabajo plantea un clasificador binario basado en VGG16 para identificar
imágenes reales y generadas por IA del conjunto CIFAKE. El informe documenta
el diseño experimental y deja identificadas las mediciones que deben
completarse después del entrenamiento; no se presuponen resultados.

## 2. Objetivo

### 2.1. General

Diseñar y evaluar un clasificador basado en VGG16 para distinguir imágenes
reales de imágenes generadas por IA utilizando una muestra balanceada de
CIFAKE.

### 2.2. Específicos

- Preparar una muestra reproducible de 10 000 imágenes, balanceada entre las
  clases REAL y FAKE.
- Separar los datos en entrenamiento, validación y prueba sin compartir
  imágenes entre particiones.
- Implementar y entrenar VGG16 para clasificación binaria.
- Evaluar el modelo en el conjunto de prueba mediante métricas de
  clasificación y una matriz de confusión.
- Analizar las limitaciones del experimento y proponer mejoras verificables.

## 3. Fundamentos teóricos

### Redes convolucionales

Una red neuronal convolucional (CNN) procesa imágenes mediante filtros
entrenables que reconocen patrones locales. Las primeras capas suelen aprender
bordes y texturas; las capas posteriores combinan esos patrones en
representaciones de mayor nivel. Las operaciones de *pooling* reducen la
resolución espacial y ayudan a controlar el costo computacional.

### Arquitectura VGG16

VGG16 es una CNN profunda propuesta por Simonyan y Zisserman. Su configuración
clásica contiene 13 capas convolucionales y 3 capas totalmente conectadas con
parámetros entrenables. Utiliza convoluciones pequeñas de 3 × 3, activaciones
ReLU y agrupamiento máximo entre bloques convolucionales.

Para clasificación binaria, la capa de salida se adapta a dos clases. Una
estrategia habitual es iniciar con pesos preentrenados en ImageNet y ajustar
las capas finales; la decisión concreta, las transformaciones y los
hiperparámetros deben registrarse cuando se implemente el entrenamiento.

### Métricas de clasificación

- **Exactitud (accuracy):** proporción total de predicciones correctas.
- **Precisión:** proporción de predicciones positivas correctas entre las
  predicciones positivas.
- **Recall (sensibilidad):** proporción de positivos reales detectados.
- **F1:** media armónica de precisión y recall.
- **Matriz de confusión:** resume aciertos y errores por clase.

Las métricas deben reportarse por clase además de en promedio, para que un
resultado global no oculte diferencias entre REAL y FAKE.

## 4. Comparación con otras arquitecturas

| Arquitectura | Característica principal | Ventaja potencial | Consideración para este caso |
|---|---|---|---|
| VGG16 | Bloques secuenciales de convoluciones 3 × 3 | Estructura clara y conocida; disponibilidad de pesos preentrenados | El clasificador original tiene muchos parámetros y puede requerir más memoria |
| ResNet | Conexiones residuales entre bloques | Facilita entrenar redes profundas | Arquitectura más compleja que VGG16 para explicar en un primer experimento |
| MobileNet | Convoluciones separables eficientes | Menor costo para inferencia en dispositivos limitados | Puede priorizar eficiencia sobre comparabilidad con el objetivo VGG16 |
| EfficientNet | Escalado coordinado de profundidad, anchura y resolución | Buen equilibrio entre tamaño y capacidad | Añade decisiones de variante y resolución al diseño experimental |

La comparación de esta tabla es cualitativa. No representa una medición
experimental sobre CIFAKE; una comparación cuantitativa requeriría entrenar y
evaluar cada arquitectura con las mismas particiones y condiciones.

## 5. Caso práctico

### 5.1. Dataset

Se utiliza CIFAKE, que contiene imágenes de la clase `real` e imágenes
sintéticas de la clase `fake`. La fuente local está organizada en
`C:\Users\USUARIO\Downloads\train\real` y
`C:\Users\USUARIO\Downloads\train\fake`, con 50 000 archivos JPG por clase
según la inspección realizada.

La fuente completa suma 100 000 imágenes. El experimento utilizará solo 10 000:
5 000 por clase, seleccionadas de forma reproducible. La fuente no se modifica
ni se copia íntegramente al repositorio.

### 5.2. Herramientas utilizadas

| Herramienta | Uso | Versión disponible en `.venv` |
|---|---|---:|
| Python | Lenguaje del proyecto | 3.13.14 |
| PyTorch | Implementación de VGG16 y entrenamiento | 2.14.1 |
| Torchvision | Arquitectura y pesos VGG16; carga de imágenes | 0.29.1 |
| Pillow | Lectura de imágenes | 13.0.0 |
| scikit-learn | Métricas de evaluación | 1.9.1 |
| Matplotlib | Curvas y matriz de confusión | 3.11.2 |
| tqdm | Progreso de las épocas | 4.70.1 |

Estas son las versiones observadas en el entorno virtual del proyecto; anotar
de nuevo las versiones usadas al ejecutar el entrenamiento final.

### 5.3. Preparación de los datos

El script `prepare_dataset.py` selecciona 5 000 imágenes por clase con semilla
42 y las copia a `data/processed/`. La división estratificada se realiza
dentro de cada clase:

| Partición | REAL | FAKE | Total |
|---|---:|---:|---:|
| Entrenamiento (80 %) | 4 000 | 4 000 | 8 000 |
| Validación (10 %) | 500 | 500 | 1 000 |
| Prueba (10 %) | 500 | 500 | 1 000 |
| **Total** | **5 000** | **5 000** | **10 000** |

La carpeta de prueba debe reservarse para la evaluación final y no utilizarse
para seleccionar hiperparámetros. Las imágenes originales permanecen en su
ubicación de origen.

### 5.4. Implementación del modelo

El modelo conserva las capas convolucionales de VGG16 preentrenadas con
ImageNet (`IMAGENET1K_V1`) y las congela. Se reemplaza el *pooling* final por
agrupamiento adaptativo global y se entrena una capa clasificadora de dos
salidas con *dropout* del 30 %. La entrada utiliza el preprocesamiento definido
por los pesos de Torchvision: redimensionamiento, recorte central y
normalización de ImageNet.

El entrenamiento usa entropía cruzada y AdamW. Se conserva el checkpoint con
mejor exactitud de validación en `models/vgg16_cifake_best.pth`. Los valores
predeterminados son 10 épocas, lote de 32, tasa de aprendizaje de 0.001 y
selección automática de CPU/GPU. Se pueden cambiar por argumentos de línea de
comandos.

### 5.5. Entrenamiento

Registrar la ejecución concreta; los valores iniciales del script son:

| Parámetro | Valor |
|---|---|
| Semilla | `[completar]` |
| Épocas | 10 (predeterminado; confirmar valor de ejecución) |
| Tamaño de lote | 32 (predeterminado; confirmar valor de ejecución) |
| Optimizador y tasa de aprendizaje | AdamW, 0.001 (predeterminado) |
| Dispositivo (CPU/GPU) | Automático; confirmar hardware usado |
| Mejor época según validación | `[completar]` |

Adjuntar las curvas de pérdida y exactitud de entrenamiento y validación en
`results/`.

### 5.6. Resultados

**Estado: pendiente de ejecutar entrenamiento y evaluación.** El script
`evaluate.py` genera el reporte JSON y la matriz de confusión. Completar con
resultados medidos en el conjunto de prueba:

| Métrica | REAL | FAKE | Promedio |
|---|---:|---:|---:|
| Precisión | `[—]` | `[—]` | `[—]` |
| Recall | `[—]` | `[—]` | `[—]` |
| F1 | `[—]` | `[—]` | `[—]` |
| Exactitud global |  |  | `[—]` |

Incluir la matriz de confusión y señalar el checkpoint y la configuración
exacta con que se obtuvieron las métricas.

### 5.7. Análisis de resultados

**Estado: pendiente.** Interpretar los errores por clase, comparar las curvas
de entrenamiento y validación, revisar posibles señales de sobreajuste y
explicar qué conclusiones permite —y cuáles no permite— obtener este conjunto
de prueba.

## 6. Ventajas y limitaciones

### Ventajas

- VGG16 es una arquitectura ampliamente documentada y relativamente sencilla
  de explicar por bloques convolucionales.
- La muestra balanceada facilita una lectura inicial de las métricas por
  clase.
- Una semilla fija facilita reproducir la selección y la división de imágenes.

### Limitaciones

- CIFAKE y una muestra de 10 000 imágenes no representan toda la diversidad
  de imágenes reales ni de generadores de IA.
- Un modelo puede aprender particularidades del conjunto de datos en vez de
  señales generales de generación sintética.
- Redimensionar imágenes pequeñas para una arquitectura preentrenada no añade
  detalle original y debe documentarse.
- Los pesos preentrenados, el hardware y la configuración de entrenamiento
  pueden afectar los resultados y la reproducibilidad.
- Las métricas no prueban por sí solas autenticidad ni deben interpretarse
  como evidencia concluyente sobre una imagen individual.

## 7. Conclusiones y recomendaciones

**Pendiente de completar con base en los resultados observados.** No redactar
conclusiones sobre precisión o desempeño antes de ejecutar la evaluación.

Recomendaciones para el análisis final:

1. Mantener el conjunto de prueba separado hasta finalizar el ajuste del
   modelo.
2. Reportar métricas por clase, matriz de confusión y variabilidad si se
   realizan varias ejecuciones.
3. Probar generalización con imágenes de fuentes no incluidas en CIFAKE.
4. Comparar arquitecturas bajo la misma división y presupuesto experimental
   antes de atribuir mejoras al modelo.
5. Presentar las predicciones como estimaciones del clasificador, no como
   veredictos sobre autenticidad.

## 8. Bibliografía

Bird, J. J., & Lotfi, A. (2023). CIFAKE: Image classification and explainable
identification of AI-generated synthetic images. *arXiv*.
https://doi.org/10.48550/arXiv.2303.14126

Krizhevsky, A., & Hinton, G. (2009). *Learning multiple layers of features
from tiny images* (Technical Report). University of Toronto.
https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

PyTorch. (s. f.). *VGG16 — Torchvision models*.
https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for
large-scale image recognition. *International Conference on Learning
Representations*. https://doi.org/10.48550/arXiv.1409.1556
