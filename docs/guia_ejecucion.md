# Guía de ejecución

Estos son los pasos que seguí para ejecutar los notebooks en Google Colab.

## 1. Preparar Google Drive

En `Mi unidad` crear esta estructura (los nombres deben coincidir):

```
Redes_Neuronales_V/
├── dataset_raw/
│   ├── persona_01/      fotos de la persona 1
│   └── persona_02/      fotos de la persona 2 (adulta, con autorización firmada)
└── fotos_demo/          2 o 3 fotos nuevas que NO estén en dataset_raw
```

Lo que tuve en cuenta al elegir las fotos:

- Un rostro principal por foto; si aparecen otras personas, recortar la foto antes de subirla.
- Variedad: distintos días, luz natural y artificial, ángulos, con y sin gafas o gorra.
- Se aceptan JPG, PNG, HEIC (iPhone) y DNG.
- El nombre de la carpeta es la etiqueta que mostrará el sistema (por ejemplo `juan_pablo`).

## 2. Ejecutar los notebooks en Colab

1. Abrir el notebook desde el botón **Open in Colab** del README.
2. *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*.
3. *Entorno de ejecución → Ejecutar todo* y autorizar el acceso a Google Drive cuando se solicite.

| Notebook | Tiempo aproximado con GPU T4 | Genera |
|---|---|---|
| 01 | 1 a 2 min | `dataset_procesado/`, figuras `01_*`, `01_registro_preprocesamiento.csv` |
| 02 | 15 a 30 min (la primera vez descarga y preprocesa LFW) | `modelos/codificador_facial.keras`, figuras `02_*`, `02_metricas_siamesa.json` |
| 03 | 1 a 3 min | figuras `03_*`, `03_metricas_reconocimiento.json` |

Si Colab se desconecta durante el Notebook 02, al volver a ejecutarlo se reutiliza la caché de LFW.

## 3. Publicar en GitHub

1. En cada notebook ya ejecutado: *Archivo → Guardar una copia en GitHub*, repositorio
   `Juanpatula2361/Taller-Redes-Neuronales-V`, ruta `notebooks/<mismo nombre>.ipynb`. Así quedan visibles las
   salidas y gráficas.
2. Subir a la carpeta `results/` del repositorio las figuras `.png` y los archivos `.json` y `.csv` de
   `Drive/Redes_Neuronales_V/results/`. Antes de subirlos, revisar que solo muestren a personas que autorizaron
   la publicación.
3. No subir `dataset_raw/`, `dataset_procesado/`, `cache/` ni `modelos/`.
