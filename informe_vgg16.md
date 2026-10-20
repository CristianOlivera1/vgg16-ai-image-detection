# Clasificación de imágenes reales y generadas por IA con VGG16

**Informe técnico del proyecto CIFAKE**

Este informe documenta el origen de los datos, la configuración del entorno y
la estrategia experimental para entrenar una VGG16 con transferencia de
aprendizaje. La primera corrida de 10 épocas está documentada con sus métricas
de entrenamiento y validación. Los resultados de prueba, y los de los
experimentos que aún no se ejecutan, permanecen deliberadamente en blanco.

> **Estado del estudio:** experimento 1 completado; evaluación final sobre
> `test` y experimentos 2 y 3 pendientes. No se atribuyen al modelo métricas
> obtenidas sobre el conjunto de prueba.

## Carátula

| Campo | Información |
|---|---|
| Institución | `[Nombre de la institución]` |
| Facultad / programa | `[Facultad o programa académico]` |
| Asignatura | `[Nombre de la asignatura]` |
| Título | Clasificación de imágenes reales y generadas por IA con VGG16 |
| Autor(es) | `[Nombre(s) del/de los estudiante(s)]` |
| Docente | `[Nombre del/de la docente]` |
| Ciudad | `[Ciudad]` |
| Fecha de entrega | `[Día, mes y año]` |

## Índice

1. [Introducción](#1-introducción)
2. [Objetivos](#2-objetivos)
   1. [Objetivo general](#21-objetivo-general)
   2. [Objetivos específicos](#22-objetivos-específicos)
3. [Fundamentos teóricos](#3-fundamentos-teóricos)
   1. [Clasificación de imágenes y CNN](#31-clasificación-de-imágenes-y-cnn)
   2. [Arquitectura VGG16](#32-arquitectura-vgg16)
   3. [Transferencia de aprendizaje](#33-transferencia-de-aprendizaje)
   4. [Pérdida y métricas](#34-pérdida-y-métricas)
4. [Comparación con otras arquitecturas](#4-comparación-con-otras-arquitecturas)
5. [Caso práctico](#5-caso-práctico)
   1. [Repositorio y organización](#51-repositorio-y-organización)
   2. [Dataset y procedencia](#52-dataset-y-procedencia)
   3. [Herramientas y entorno](#53-herramientas-y-entorno)
   4. [Preparación y partición de los datos](#54-preparación-y-partición-de-los-datos)
   5. [Configuración del modelo](#55-configuración-del-modelo)
   6. [Configuración y uso de la GPU](#56-configuración-y-uso-de-la-gpu)
   7. [Diseño de experimentos](#57-diseño-de-experimentos)
   8. [Experimento 1: clasificador congelado, 10 épocas](#58-experimento-1-clasificador-congelado-10-épocas)
   9. [Experimento 2: clasificador congelado, 15 épocas](#59-experimento-2-clasificador-congelado-15-épocas)
   10. [Resultados de prueba](#510-resultados-de-prueba)
   11. [Interpretación de las corridas](#511-interpretación-de-las-corridas)
6. [Ventajas, limitaciones y amenazas a la validez](#6-ventajas-limitaciones-y-amenazas-a-la-validez)
7. [Conclusiones y recomendaciones](#7-conclusiones-y-recomendaciones)
8. [Bibliografía](#8-bibliografía)

## 1. Introducción

Los sistemas generativos pueden crear imágenes de apariencia realista, lo que
complica distinguir imágenes sintéticas de imágenes procedentes de fotografías
reales. Una red neuronal convolucional puede aprender patrones visuales que
ayuden a clasificar ejemplos, pero el desempeño medido en un conjunto de datos
no garantiza que el detector funcione igual de bien ante otros generadores,
calidades, formatos o dominios.

Este proyecto estudia una clasificación binaria con VGG16 y CIFAKE. La fuente
contiene imágenes reales y sintéticas; para que el experimento sea manejable,
reproducible y balanceado se seleccionan 10 000 imágenes y se separan en
entrenamiento, validación y prueba. La primera estrategia congela las capas
convolucionales preentrenadas en ImageNet y entrena una nueva cabeza
clasificadora.

El informe diferencia entre **resultados de validación**, usados para observar
el aprendizaje y seleccionar un checkpoint, y **resultados de prueba**, que
deben calcularse una sola vez después de cerrar las decisiones del modelo. La
exactitud de validación de la primera corrida es 87,20 % en la época 10; esta
cifra no debe interpretarse como exactitud de prueba ni como una medida
universal para detectar cualquier imagen generada por IA.

## 2. Objetivos

### 2.1. Objetivo general

Implementar y estudiar un clasificador basado en VGG16 para distinguir las
clases REAL y FAKE del conjunto CIFAKE mediante transferencia de aprendizaje y
una evaluación experimental separada por particiones.

### 2.2. Objetivos específicos

- Seleccionar de forma reproducible 5 000 imágenes de cada clase.
- Crear particiones balanceadas de entrenamiento, validación y prueba en
  proporción 80/10/10.
- Entrenar una cabeza binaria sobre las características congeladas de VGG16.
- Usar la validación para seguir el aprendizaje y elegir el mejor checkpoint,
  manteniendo test fuera del ciclo de ajuste.
- Comparar estrategias de entrenamiento de forma controlada, cambiando un
  factor relevante a la vez.
- Reportar métricas por clase, matriz de confusión, configuración de hardware
  y limitaciones de generalización.

## 3. Fundamentos teóricos

### 3.1. Clasificación de imágenes y CNN

Una imagen puede representarse como un arreglo de píxeles y canales de color.
Una red neuronal convolucional (CNN) aplica filtros compartidos sobre regiones
locales para aprender patrones espaciales. Las capas iniciales suelen
responder a bordes, colores y texturas; las capas más profundas combinan esas
respuestas en representaciones de mayor nivel. Las capas de *pooling* reducen
la dimensión espacial de las activaciones.

En clasificación supervisada, cada imagen se asocia con una etiqueta conocida.
El modelo genera puntuaciones para las clases y sus parámetros se optimizan
para reducir una función de pérdida sobre los ejemplos etiquetados.

### 3.2. Arquitectura VGG16

VGG16 es una arquitectura convolucional profunda presentada por Simonyan y
Zisserman. Su configuración clásica organiza 13 capas convolucionales en
bloques y tres capas totalmente conectadas. Emplea filtros de 3 × 3, ReLU y
*max pooling*. La estructura secuencial hace que el flujo de procesamiento sea
relativamente fácil de describir y visualizar.

La implementación de este proyecto parte de VGG16 de Torchvision con pesos
`IMAGENET1K_V1`. Para adaptar la salida al problema binario, se sustituyen el
*pooling* espacial final por `AdaptiveAvgPool2d((1, 1))` y el clasificador
original por *dropout* de 0,3 seguido de una capa lineal de 512 características
a dos logits: FAKE y REAL.

### 3.3. Transferencia de aprendizaje

La transferencia de aprendizaje reutiliza representaciones aprendidas en una
tarea con muchos datos para iniciar otra tarea relacionada. En el primer
experimento se congelan todos los parámetros convolucionales de VGG16 y solo
se actualiza la cabeza clasificadora. Esto:

- reduce la cantidad de parámetros que se entrenan;
- disminuye el costo frente a ajustar toda la red;
- proporciona una línea base antes de probar el ajuste fino;
- reduce, aunque no elimina, el riesgo de sobreajustar una muestra limitada.

Una segunda estrategia razonable es descongelar el último bloque convolucional
y ajustarlo con una tasa de aprendizaje pequeña. Es una hipótesis de mejora,
no un resultado anticipado: sus métricas se deben medir con el mismo conjunto
de datos y el mismo protocolo.

### 3.4. Pérdida y métricas

La pérdida usada es entropía cruzada multiclase (`CrossEntropyLoss`) sobre los
dos logits. El optimizador es AdamW. La exactitud se calcula como la proporción
de etiquetas correctamente clasificadas.

En la evaluación final se reportarán además:

- **Precisión por clase:** de las predicciones de una clase, qué proporción
  corresponde realmente a ella.
- **Recall por clase:** de los ejemplos reales de una clase, qué proporción
  recupera el modelo.
- **F1 por clase:** media armónica de precisión y recall.
- **Promedio macro:** promedio no ponderado de la métrica de las dos clases.
- **Matriz de confusión:** conteo de predicciones frente a etiquetas reales;
  permite distinguir los falsos positivos de los falsos negativos.

La exactitud sola puede ocultar errores asimétricos, por lo que no será la única
métrica de la evaluación de prueba.

## 4. Comparación con otras arquitecturas

La siguiente comparación es **teórica y cualitativa**; no se han entrenado
estos modelos en este proyecto. Se incluyen arquitecturas representativas para
dar contexto a VGG16 y porque ilustran compromisos diferentes de profundidad,
eficiencia y complejidad.

| Arquitectura | Idea principal | Razón para compararla | Consideración en este proyecto |
|---|---|---|---|
| **VGG16** | Bloques secuenciales de convoluciones 3 × 3 | Es el modelo requerido y una referencia didáctica clara para transferencia de aprendizaje | Consume más cómputo que arquitecturas modernas eficientes |
| **ResNet** | Conexiones residuales que facilitan el flujo de gradientes | Representa una alternativa profunda ampliamente usada y permite estudiar el efecto de conexiones *skip* | Añade una estructura más compleja que la secuencia de VGG |
| **MobileNet** | Convoluciones separables para reducir costo | Permite contrastar con un diseño orientado a dispositivos con recursos limitados | Busca eficiencia y no necesariamente iguala el objetivo pedagógico de VGG16 |
| **EfficientNet** | Escalado coordinado de resolución, profundidad y anchura | Ilustra una familia moderna que busca equilibrar tamaño y capacidad | Requiere escoger una variante y controlar también la resolución |

Una comparación experimental válida debe entrenar cada arquitectura con la
misma división de datos, preprocesamiento comparable, criterio de selección,
presupuesto de ajuste y conjunto de prueba. Hasta que se realice ese
experimento, no se atribuye superioridad cuantitativa a ninguna arquitectura.

## 5. Caso práctico

### 5.1. Repositorio y organización

El código del proyecto está en el repositorio:
[github.com/CristianOlivera1/vgg16-ai-image-detection](https://github.com/CristianOlivera1/vgg16-ai-image-detection).

| Ruta | Responsabilidad |
|---|---|
| `prepare_dataset.py` | Selecciona, balancea y separa imágenes del origen local |
| `data/processed/` | Particiones de imágenes organizadas para `ImageFolder` |
| `model_utils.py` | Constructor de VGG16, clases, dispositivo y checkpoints |
| `train.py` | Entrenamiento y validación; guarda el mejor checkpoint y curvas |
| `evaluate.py` | Evaluación final en `test`, reporte JSON y matriz de confusión |
| `predict.py` | Predicción individual con un checkpoint existente |
| `models/` | Checkpoints locales; no son resultados de prueba |
| `results/` | Historiales y gráficas generados por las ejecuciones |
| `docs/figures/` | Imagen de la primera corrida incluida en este informe |

Las imágenes del dataset y los pesos del modelo no se deben incluir en Git.
El repositorio contiene código y documentación; el dataset se obtiene y se
mantiene localmente.

### 5.2. Dataset y procedencia

La fuente de datos es
[CIFAKE en Hugging Face](https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible).
La ficha del dataset describe 50 000 imágenes REAL y 50 000 FAKE. Las imágenes
reales provienen de CIFAR-10 y las imágenes sintéticas fueron creadas con
Stable Diffusion v1.4. La copia alojada en Hugging Face elimina de la
distribución las imágenes de los splits de prueba originales, conservando las
imágenes de entrenamiento del dataset original.

En este entorno, la copia local examinada está en
`C:\Users\USUARIO\Downloads\train`, con 50 000 JPG directamente en cada una de
las carpetas `real/` y `fake/`. Una imagen inspeccionada por clase mide 32 × 32
píxeles, está en modo RGB y es JPEG.

La ficha de Hugging Face enlaza el trabajo de Bird y Lotfi (2023) y declara la
licencia MIT para la copia del conjunto, con atribución a CIFAR-10 y a los
autores de CIFAKE. Consultar la ficha y conservar las citas pertinentes si se
redistribuyen los datos o se publica el trabajo.

### 5.3. Herramientas y entorno

| Componente | Configuración observada |
|---|---|
| Sistema de ejecución | Windows, PowerShell y entorno virtual `.venv` |
| Python | 3.13.14 |
| PyTorch | 2.11.0+cu128 |
| Torchvision | 0.26.0+cu128 |
| Runtime CUDA de PyTorch | 12.8 |
| cuDNN | 91900 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Memoria GPU detectada | 6,0 GiB |
| Pillow | 13.0.0 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| tqdm | 4.70.1 |

Las versiones de PyTorch y Torchvision anteriores corresponden al entorno
virtual revisado. Para una reproducción estricta, guardar también la salida
de `python -m pip freeze` del mismo entorno con el que se ejecute cada
experimento.

### 5.4. Preparación y partición de los datos

El preparador toma una muestra aleatoria sin reemplazo de 5 000 archivos por
clase usando la semilla 42. A continuación mezcla cada lista y copia los
archivos a `data/processed/`; no mueve, modifica ni incorpora la fuente al
repositorio. El orden alfabético inicial de los nombres y el muestreo con
semilla fija permiten repetir la selección mientras la fuente no cambie.

#### Justificación de la proporción 80/10/10

La asignación 80/10/10 es una decisión práctica para este experimento, no una
regla universal:

1. **Entrenamiento (80 %, 8 000 imágenes):** conserva la mayor parte de la
   muestra para ajustar la cabeza del modelo.
2. **Validación (10 %, 1 000 imágenes):** ofrece ejemplos de ambas clases para
   observar la evolución entre épocas y seleccionar el checkpoint, sin usar
   las etiquetas de test para ajustar el modelo.
3. **Prueba (10 %, 1 000 imágenes):** mantiene una partición independiente
   para la evaluación final después de congelar las decisiones de diseño.

Al balancear las clases dentro de cada partición, accuracy tiene una
interpretación inicial más directa y ambas clases tienen representación
suficiente para métricas por clase. Una partición mayor de test aumentaría el
número de ejemplos para evaluación a costa del entrenamiento; una validación
muy pequeña haría más variable la selección del checkpoint.

| Partición | REAL | FAKE | Total | Uso permitido |
|---|---:|---:|---:|---|
| `train` | 4 000 | 4 000 | 8 000 | Ajustar parámetros |
| `val` | 500 | 500 | 1 000 | Comparar épocas/configuraciones y elegir checkpoint |
| `test` | 500 | 500 | 1 000 | Una evaluación final después de cerrar los experimentos |
| **Total** | **5 000** | **5 000** | **10 000** | |

El directorio de test creado aquí es una partición nueva extraída del contenido
local de `train`; **no es el test oficial original de CIFAKE**. Debido a que la
copia de Hugging Face ya excluyó los ejemplos del test original, esta
evaluación permite estimar el rendimiento en imágenes retenidas de la misma
distribución de entrenamiento, pero no demuestra generalización a imágenes
independientes de otros generadores o fuentes.

### 5.5. Configuración del modelo

| Decisión | Configuración de la corrida base |
|---|---|
| Arquitectura | VGG16 de Torchvision |
| Pesos iniciales | `IMAGENET1K_V1` |
| Backbone | Capas convolucionales congeladas |
| Cabeza | `AdaptiveAvgPool2d((1, 1))`, `Dropout(0.3)`, `Linear(512, 2)` |
| Orden de etiquetas | `FAKE = 0`, `REAL = 1` |
| Transformación | Preprocesamiento del preset de pesos de Torchvision |
| Entrada al modelo | Redimensionamiento a 256 y recorte central a 224 × 224 |
| Normalización | Media y desviación estándar de ImageNet definidas por el preset |
| Función de pérdida | Entropía cruzada |
| Optimizador | AdamW |
| Tasa de aprendizaje | 0.001 |
| Selección de checkpoint | Mayor accuracy de validación |
| Planificador de tasa | No configurado en esta versión |
| Parada temprana | No configurada; se completan las épocas solicitadas |

Los parámetros congelados no reciben gradientes. El entrenamiento actualiza
solamente los pesos y sesgos de la capa lineal de la cabeza. Aunque VGG16
preentrenada fue diseñada para imágenes de mayor resolución, los JPG de CIFAKE
son de 32 × 32; el redimensionamiento a 224 × 224 es necesario para el preset
de ImageNet, pero no recupera detalle que no estaba en la imagen original.

### 5.6. Configuración y uso de la GPU

Se activó el entorno virtual y se indicó CUDA explícitamente:

```powershell
.\.venv\Scripts\Activate.ps1
python train.py --device cuda
```

`--device cuda` pide ejecutar el modelo y los lotes en CUDA; el programa falla
si CUDA no está disponible, en vez de fingir una ejecución en GPU. Para la
corrida registrada, la consola imprimió `Dispositivo: cuda`, y el hardware
reportado fue la NVIDIA GeForce RTX 3050 Laptop GPU con 6 GB de VRAM. PyTorch
detectó el runtime CUDA 12.8 y cuDNN 91900 en la inspección del entorno.

La semilla predeterminada es 42. El script inicializa los generadores de
PyTorch y CUDA, habilita determinismo de cuDNN y deshabilita
`cudnn.benchmark`. Los lotes se transfieren al dispositivo; en CUDA el
`DataLoader` utiliza `pin_memory=True`. El tamaño de lote predeterminado es 32
y el número de workers es 0. No se habilitó entrenamiento de precisión mixta
(AMP) en la corrida base.

La misma corrida también puede iniciarse de forma reproducible explícita con:

```powershell
python train.py --device cuda --epochs 10 --batch-size 32 `
  --learning-rate 0.001 --workers 0 --seed 42
```

### 5.7. Diseño de experimentos

El plan compara estrategias para entender qué aporta cada cambio. La corrida
de 10 épocas ya ejecutada es la línea base. Una repetición con 15 épocas prueba
si prolongar el mismo entrenamiento aporta mejoras; el ajuste fino del último
bloque prueba si actualizar características convolucionales más específicas
ayuda a esta tarea. En cada comparación se mantendrán iguales los datos, las
transformaciones y la semilla, salvo la variable que se estudia.

| Experimento | Estrategia | Cambio respecto a la línea base | Estado |
|---|---|---|---|
| 1 | VGG16 congelada, 10 épocas | Línea base; clasificador lineal nuevo | **Completado** |
| 2 | VGG16 congelada, 15 épocas | Cambiar solo el número total de épocas; iniciar una nueva corrida | **Completado** |
| 3 | Ajuste fino del último bloque VGG16 | Descongelar el último bloque y usar una tasa menor para el backbone; probar 5–8 épocas y lote 16 | Pendiente; requiere ampliar el código de entrenamiento |

**Las corridas no se reanudan automáticamente.** `train.py --epochs 15`
empieza una ejecución nueva desde los pesos ImageNet y una cabeza recién
inicializada; no significa continuar 15 épocas desde el checkpoint anterior.
Asimismo, el script guarda el mejor checkpoint en el mismo nombre
`models/vgg16_cifake_best.pth`, y reemplaza ese archivo cuando encuentra una
accuracy de validación mayor. Para conservar resultados y checkpoints, copiar
los artefactos con un nombre que identifique cada experimento antes de la
siguiente corrida.

La estrategia 3 no consiste solo en cambiar la opción `--epochs`: hay que
modificar el constructor y el optimizador para descongelar las capas elegidas,
asignar una tasa de aprendizaje menor al backbone y registrar claramente esos
valores. No se debe presentar como ejecutada hasta implementar y medir esa
variante.

### 5.8. Experimento 1: clasificador congelado, 10 épocas

- **Identificador:** `frozen_vgg16_10ep`
- **Estado:** ejecutado según la salida de consola y los artefactos guardados.
- **Comando:** `python train.py --device cuda`
- **Configuración:** 10 épocas, lote 32, AdamW, tasa 0.001, semilla 42 y
  clasificador nuevo sobre VGG16 ImageNet congelada.
- **Hardware informado:** NVIDIA GeForce RTX 3050 6GB Laptop GPU.

#### Métricas por época

| Época | Pérdida train | Accuracy train | Pérdida val | Accuracy val |
|---:|---:|---:|---:|---:|
| 1 | 0.5422 | 74.84 % | 0.4596 | 81.00 % |
| 2 | 0.4359 | 81.76 % | 0.4069 | 83.50 % |
| 3 | 0.4086 | 82.80 % | 0.3807 | 84.60 % |
| 4 | 0.3887 | 83.50 % | 0.3648 | 85.20 % |
| 5 | 0.3783 | 83.81 % | 0.3563 | 84.90 % |
| 6 | 0.3690 | 83.96 % | 0.3480 | 85.90 % |
| 7 | 0.3674 | 84.38 % | 0.3392 | 86.20 % |
| 8 | 0.3658 | 83.81 % | 0.3359 | 86.40 % |
| 9 | 0.3621 | 84.30 % | 0.3311 | 86.50 % |
| 10 | 0.3604 | 84.21 % | 0.3281 | 87.20 % |

La mejor accuracy de validación observada es **87,20 %**, en la época 10.
También en la época 10 se observa la menor pérdida de validación de esta
corrida, **0,3281**. El checkpoint fue seleccionado por accuracy de
validación; ambos criterios coinciden en la última época en esta ejecución.
Estos son resultados de validación, no de test.

#### Curvas de aprendizaje

La figura presenta la pérdida y exactitud de entrenamiento y validación para
las 10 épocas. Se incluye una copia en el repositorio para que la imagen se
muestre al visualizar este informe en GitHub; la salida original de
entrenamiento está en `results/loss_accuracy_frozen_10ep.png`.

![Curvas de pérdida y exactitud del experimento 1: VGG16 congelada durante 10 épocas](docs/figures/loss_accuracy_frozen_10ep.png)

**Figura 1.** Curvas de entrenamiento y validación de
`frozen_vgg16_10ep`. Fuente: `results/training_history_frozen_10ep.csv`.

#### Artefactos de esta corrida

| Artefacto | Ruta local | Contenido |
|---|---|---|
| Checkpoint conservado | `models/vgg16_cifake_frozen_10ep.pth` | Mejor checkpoint de la corrida base |
| Historial CSV | `results/training_history_frozen_10ep.csv` | Pérdidas y accuracies por época |
| Figura de la corrida | `results/loss_accuracy_frozen_10ep.png` | Curvas generadas por el entrenamiento |
| Figura incluida en el informe | `docs/figures/loss_accuracy_frozen_10ep.png` | Copia para visualizar el informe en GitHub |

Los checkpoints contienen pesos del modelo; **no contienen métricas de test**.
El archivo CSV es la fuente numérica de la figura y de la tabla por época.

### 5.9. Experimento 2: clasificador congelado, 15 épocas

- **Identificador:** `frozen_vgg16_15ep`
- **Estado:** completado según la salida de consola y el historial guardado.
- **Comando:** `python train.py --device cuda --epochs 15`
- **Configuración:** mismo modelo y valores de la corrida base; se aumentó el
  total de 10 a 15 épocas.
- **Hardware informado:** NVIDIA GeForce RTX 3050 6GB Laptop GPU.

#### Métricas por época

| Época | Pérdida train | Accuracy train | Pérdida val | Accuracy val |
|---:|---:|---:|---:|---:|
| 1 | 0.5422 | 74.84 % | 0.4596 | 81.00 % |
| 2 | 0.4359 | 81.76 % | 0.4069 | 83.50 % |
| 3 | 0.4086 | 82.80 % | 0.3807 | 84.60 % |
| 4 | 0.3887 | 83.50 % | 0.3648 | 85.20 % |
| 5 | 0.3783 | 83.81 % | 0.3563 | 84.90 % |
| 6 | 0.3690 | 83.96 % | 0.3480 | 85.90 % |
| 7 | 0.3674 | 84.38 % | 0.3392 | 86.20 % |
| 8 | 0.3658 | 83.81 % | 0.3359 | 86.40 % |
| 9 | 0.3621 | 84.30 % | 0.3311 | 86.50 % |
| 10 | 0.3604 | 84.21 % | 0.3281 | 87.20 % |
| 11 | 0.3564 | 84.36 % | 0.3258 | 86.50 % |
| 12 | 0.3513 | 84.76 % | 0.3253 | 86.20 % |
| 13 | 0.3490 | 84.84 % | 0.3221 | 86.20 % |
| 14 | 0.3506 | 84.69 % | 0.3149 | **87.60 %** |
| 15 | 0.3481 | 85.19 % | **0.3137** | 87.40 % |

El máximo de accuracy de validación fue **87,60 % en la época 14**, frente a
87,20 % en la época 10 de la corrida base: una mejora de **0,40 puntos
porcentuales** en validación. La pérdida de validación más baja se obtuvo en la
época 15 (**0,3137**), aunque su accuracy (**87,40 %**) fue ligeramente menor
que el máximo de la época 14. El código conserva el checkpoint con mejor
accuracy de validación, por lo que el checkpoint de esta corrida corresponde
a la época 14 y no a la época 15.

Las métricas de las épocas 1 a 10 coinciden con las de la corrida base. Esto es
coherente con que ambas ejecuciones partieron de nuevo con la misma semilla,
datos, pesos preentrenados y configuración; la ejecución de 15 épocas no fue
una continuación del checkpoint de 10 épocas. No se interpreta esta
coincidencia como una prueba de variabilidad entre semillas, ya que la semilla
se mantuvo fija.

#### Curvas y artefactos

![Curvas de pérdida y exactitud del experimento 2: VGG16 congelada durante 15 épocas](docs/figures/loss_accuracy_frozen_15ep.png)

**Figura 2.** Curvas de entrenamiento y validación del experimento
`frozen_vgg16_15ep`. Fuente: `results/training_history_frozen_15ep.csv`.

| Artefacto | Ruta local | Contenido |
|---|---|---|
| Checkpoint conservado | `models/vgg16_cifake_frozen_15ep.pth` | Mejor checkpoint de esta corrida; época 14 según accuracy de validación |
| Historial CSV | `results/training_history_frozen_15ep.csv` | Pérdidas y accuracies de las 15 épocas |
| Figura de la corrida | `results/loss_accuracy_frozen_15ep.png` | Curvas generadas por el entrenamiento |
| Figura incluida en el informe | `docs/figures/loss_accuracy_frozen_15ep.png` | Copia para visualizar el informe en GitHub |

### 5.10. Resultados de prueba

**Pendiente.** No se ha consignado aquí ninguna evaluación con `evaluate.py`.
Una vez seleccionada la estrategia final, ejecutar:

```powershell
python evaluate.py --device cuda
```

Completar esta sección con el reporte producido sobre `data/processed/test`.
No usar estos resultados para decidir retrospectivamente cuál experimento
preferir; de ser necesario elegir modelos, hacerlo con validación antes de
abrir test.

| Métrica de test | REAL | FAKE | Promedio macro |
|---|---:|---:|---:|
| Precisión |  |  |  |
| Recall |  |  |  |
| F1 |  |  |  |
| Accuracy global |  |  |  |

**Matriz de confusión (completar con `results/metrics_report.json`):**

| Etiqueta real \ predicción | FAKE | REAL |
|---|---:|---:|
| FAKE |  |  |
| REAL |  |  |

- **Checkpoint evaluado:**
- **Fecha de evaluación:**
- **Observaciones:**

### 5.11. Interpretación de las corridas

En ambas corridas las pérdidas de entrenamiento y validación disminuyen en
general. La corrida de 15 épocas alcanzó una accuracy de validación máxima
ligeramente mayor en la época 14 que la corrida base en la época 10. La
accuracy de validación no aumentó de forma monótona: tras 87,20 % en la época
10 bajó durante las épocas 11–13, llegó a 87,60 % en la 14 y terminó en
87,40 %. Por tanto, seleccionar el checkpoint por validación es importante y
la época final no necesariamente es la mejor según accuracy.

La pérdida de validación continuó bajando hasta la época 15. La diferencia
entre el mínimo de pérdida (época 15) y el máximo de accuracy (época 14)
muestra que los criterios no son intercambiables: la regla actual de guardar
el mejor checkpoint usa accuracy de validación. La mejora de 0,40 puntos
porcentuales frente a la línea base es una mejora observada sobre `val`, no
evidencia de mejora en `test`.

La repetición de las métricas iniciales de la época 1 a la 10 concuerda con la
semilla fija y la ejecución determinista configurada; no permite medir la
estabilidad ante inicializaciones distintas. Una comparación de variabilidad
requiere entrenar con otras semillas y registrar cada corrida.

La accuracy de entrenamiento es inferior a la de validación en este
experimento. Una explicación consistente con la implementación es que
`Dropout(0.3)` está activo al entrenar y desactivado al validar; el modelo
entrenado enfrenta ruido de regularización que no aparece en evaluación. Esta
diferencia no prueba por sí misma un problema ni una mejor generalización.

El experimento 3 estudiará si adaptar el último bloque convolucional aporta
información útil para CIFAKE. Esta variante todavía requiere modificar el
entrenamiento y medirla; no es un resultado existente. Ninguna observación de
validación sustituye la evaluación final en test.

## 6. Ventajas, limitaciones y amenazas a la validez

### Ventajas

- La selección balanceada evita que una clase domine por frecuencia en la
  muestra usada.
- La semilla fija hace reproducible el muestreo mientras no cambie la fuente.
- Congelar el backbone crea una línea base de menor costo antes del ajuste
  fino.
- Validación y prueba se mantienen separadas para reducir la contaminación
  del ciclo de desarrollo.
- PyTorch, Torchvision, el repositorio y la fuente de datos pueden ser
  consultados por terceros.

### Limitaciones

- El conjunto contiene imágenes pequeñas de 32 × 32; redimensionarlas no crea
  información visual nueva.
- VGG16 consume más memoria y cómputo que modelos diseñados para inferencia
  eficiente.
- Una única partición y una única semilla no miden la variabilidad del
  entrenamiento.
- Los datos no representan todos los generadores de imágenes ni todas las
  fuentes fotográficas.
- La clase REAL procede de CIFAR-10 y la clase FAKE de Stable Diffusion v1.4;
  un modelo podría aprender diferencias propias de esas fuentes en lugar de
  señales universales de síntesis.
- La muestra de test local proviene del conjunto de entrenamiento publicado,
  ya que esta copia del dataset no contiene el test original. No es una prueba
  de generalización externa.
- Una predicción del clasificador es una estimación y no certifica la
  autenticidad de una imagen.

### Amenazas a la validez y mitigaciones

| Amenaza | Consecuencia | Mitigación prevista |
|---|---|---|
| Sobreajuste a la distribución CIFAKE | Métricas altas en CIFAKE podrían no transferirse a imágenes nuevas | Evaluar posteriormente con fuentes externas y generadores no vistos |
| Elección repetida según validación | Sobreajuste indirecto a `val` tras muchas pruebas | Registrar experimentos y reservar `test` hasta elegir el modelo |
| Variabilidad aleatoria | Una sola semilla puede dar una estimación inestable | Repetir la estrategia final con varias semillas si el alcance lo permite |
| Fuga entre particiones | Duplicados en train/val/test inflarían resultados | Usar muestreo sin reemplazo dentro de cada clase y revisar duplicados si se amplía el estudio |
| Desbalance de métricas | Accuracy no revela errores por clase | Reportar precisión, recall, F1 y matriz de confusión |
| Diferencias de dominio | Los artefactos del origen pueden confundirse con la clase | Interpretar el resultado como específico de CIFAKE y probar dominio externo |

## 7. Conclusiones y recomendaciones

### Conclusiones provisionales

Las dos corridas confirman que la cabeza clasificadora sobre VGG16
preentrenada aprende sobre la muestra CIFAKE. La corrida de 10 épocas alcanzó
87,20 % de accuracy de validación; la de 15 alcanzó 87,60 % en la época 14.
La mínima pérdida de validación de la corrida larga se observó en la época 15.
Todas estas cifras corresponden exclusivamente a `val`.

No se puede concluir aún que el modelo logre 87,20 % en datos de prueba o que
detecte imágenes generadas por sistemas diferentes a Stable Diffusion v1.4.
La conclusión definitiva depende de la evaluación retenida, del análisis de
errores y de las comparaciones que realmente se ejecuten.

### Recomendaciones para continuar

1. Las copias del checkpoint, CSV y figuras de las corridas 10 y 15 épocas ya
   se conservaron con identificadores específicos.
2. Implementar el experimento de ajuste fino por separado; usar una tasa
   pequeña para el bloque descongelado y documentar las tasas de cada grupo de
   parámetros.
3. Elegir la configuración con validación y evaluar test una sola vez al
   finalizar las comparaciones.
4. Para ampliar el estudio, repetir con varias semillas y evaluar imágenes
   externas a CIFAKE.
5. Documentar cualquier cambio de arquitectura, lote, tasa de aprendizaje,
   resolución, pesos iniciales o preprocesamiento como un experimento nuevo.

### Registro para experimentos futuros

Rellenar una fila por corrida y conservar sus artefactos con identificadores
distintos. Los resultados de estas corridas se dejan en blanco hasta medirlos.

| Experimento | Épocas | Estrategia / cambio | Mejor época | Mejor accuracy val | Pérdida val | Accuracy test | Checkpoint / figura |
|---|---:|---|---:|---:|---:|---:|---|
| 1 | 10 | VGG16 congelada; clasificador lineal | 10 | 87,20 % | 0,3281 |  | `frozen_10ep` |
| 2 | 15 | VGG16 congelada; cambia solo duración | 14 | 87,60 % | 0,3137 (época 15) |  | `frozen_15ep` |
| 3 | 5–8 | Ajuste fino del último bloque; tasa menor y lote 16 |  |  |  |  |  |

## 8. Bibliografía

Bird, J. J., & Lotfi, A. (2023). CIFAKE: Image classification and explainable
identification of AI-generated synthetic images. *arXiv*.
https://doi.org/10.48550/arXiv.2303.14126

Hugging Face. (s. f.). *CIFAKE_autotrain_compatible* [Conjunto de datos].
https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T.,
Andreetto, M., & Adam, H. (2017). MobileNets: Efficient convolutional neural
networks for mobile vision applications. *arXiv*.
https://doi.org/10.48550/arXiv.1704.04861

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. In *Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition* (pp. 770–778).
https://doi.org/10.1109/CVPR.2016.90

Krizhevsky, A., & Hinton, G. (2009). *Learning multiple layers of features
from tiny images* (Technical Report). University of Toronto.
https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

PyTorch. (s. f.). *VGG16 — Torchvision models*.
https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html

Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for
large-scale image recognition. *International Conference on Learning
Representations*. https://doi.org/10.48550/arXiv.1409.1556

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for
convolutional neural networks. *arXiv*.
https://doi.org/10.48550/arXiv.1905.11946
