# Detección de imágenes generadas por IA con VGG16

Proyecto de clasificación binaria con VGG16 para distinguir imágenes reales de
imágenes generadas por IA. El dataset fuente CIFAKE se conserva en su ubicación
original; el proyecto copia una muestra balanceada de 10 000 imágenes y la
divide en entrenamiento, validación y prueba.

## Estructura

```text
vgg16-ai-image-detection/
├── data/
│   ├── raw/                 # Referencia/documentación del dataset fuente
│   └── processed/           # Se crea al ejecutar prepare_dataset.py
│       ├── train/REAL/
│       ├── train/FAKE/
│       ├── val/REAL/
│       ├── val/FAKE/
│       ├── test/REAL/
│       └── test/FAKE/
├── models/                  # Pesos/checkpoints locales; no se versionan
├── results/                 # Métricas y gráficas generadas
├── prepare_dataset.py
├── train.py
├── evaluate.py
├── predict.py
├── model_utils.py
├── requirements.txt
└── informe_vgg16.md
```

## Preparar los datos

La fuente esperada contiene `real/` y `fake/`, con imágenes JPG directamente
dentro de cada carpeta. La configuración predeterminada apunta a
`C:\Users\USUARIO\Downloads\train`.

```powershell
.\.venv\Scripts\Activate.ps1
python prepare_dataset.py
```

Para usar otra ubicación:

```powershell
python prepare_dataset.py --source-dir "D:\datasets\CIFAKE\train"
```

El muestreo usa la semilla `42`, selecciona 5 000 imágenes por clase y copia,
sin mover ni modificar la fuente, 4 000 por clase a `train`, 500 por clase a
`val` y 500 por clase a `test`. El destino no se sobrescribe: para regenerarlo,
renombra o elimina primero `data/processed`.

## Entrenar, evaluar y predecir

```powershell
python -m pip install -r requirements.txt
python train.py
python evaluate.py
python predict.py "C:\ruta\a\imagen.jpg"
```

El entrenamiento usa VGG16 preentrenado en ImageNet y ajusta un clasificador
binario. Por defecto se congelan las capas convolucionales; la estrategia
`finetune-last-block` permite entrenar el bloque final con una tasa menor.
`--device auto` selecciona CUDA si está disponible y, en caso contrario, CPU.
En el primer entrenamiento Torchvision puede descargar los pesos preentrenados.

Para ajustar el último bloque convolucional con tasas de aprendizaje
separadas y guardar la ejecución en archivos propios:

```powershell
python train.py --device cuda --strategy finetune-last-block `
  --epochs 8 --batch-size 16 --learning-rate 1e-4 `
  --backbone-learning-rate 1e-5 --seed 42 `
  --run-name vgg16_cifake_finetune_block5_8ep
```

Artefactos generados:

- `models/vgg16_cifake_best.pth`: checkpoint con mejor exactitud de validación.
- `results/training_history.csv` y `results/loss_accuracy_curves.png`:
  historial y curvas de entrenamiento.
- `results/metrics_report.json` y `results/confusion_matrix.png`:
  evaluación final en test.

Se pueden ajustar épocas, lote, tasa de aprendizaje, workers y dispositivo;
por ejemplo: `python train.py --epochs 5 --batch-size 16 --device cpu`.
Los directorios de datos, modelos y resultados se excluyen de Git.
