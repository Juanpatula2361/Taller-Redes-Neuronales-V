# Taller Redes Neuronales V — Reconocimiento facial con redes siamesas y Triplet Loss

**Universidad de Cundinamarca · Especialización en Inteligencia Artificial · Deep Learning – Conceptos (2026)**
**Actividad:** R2-A1-S10 Redes Neuronales V (semana 10)
**Autor:** Juan Pablo Cristancho Tula

En este repositorio dejo el trabajo de la semana 10, dedicada al reconocimiento de rostros. Implementé en
**TensorFlow + Keras** una red neuronal siamesa que convierte cada cara en un vector de 128 números y decide si dos
fotos son de la misma persona midiendo la distancia entre esos vectores.

El modelo lo entrené con **Triplet Loss** sobre LFW, un conjunto público de rostros, y después lo probé, sin volver a
entrenarlo, con un dataset propio: fotos mías y de una persona adulta que autorizó su uso académico. Con ese dataset
hice pruebas de **one-shot** (reconocer con una sola foto), **few-shot** (con pocas fotos) y **zero-shot**
(reconocer a personas que el modelo nunca vio y rechazar a quien no está registrado).

## Notebooks

| Notebook | Qué hago en él | Colab |
|---|---|---|
| `01_preprocesamiento_dataset_facial` | Cargo las fotos (JPG, PNG, HEIC, DNG), corrijo la orientación, detecto la cara con MTCNN, la alineo por los ojos, la recorto a 160×160 y reviso la calidad | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Juanpatula2361/Taller-Redes-Neuronales-V/blob/main/notebooks/01_preprocesamiento_dataset_facial.ipynb) |
| `02_red_siamesa_triplet_loss` | Construyo la red siamesa, implemento Triplet Loss, entreno en dos fases y evalúo con ROC, AUC y EER frente a una línea base | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Juanpatula2361/Taller-Redes-Neuronales-V/blob/main/notebooks/02_red_siamesa_triplet_loss.ipynb) |
| `03_reconocimiento_one_few_zero_shot` | Pruebo el reconocimiento con mi dataset: matriz de distancias, one-shot, few-shot, zero-shot y una demostración con fotos nuevas | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Juanpatula2361/Taller-Redes-Neuronales-V/blob/main/notebooks/03_reconocimiento_one_few_zero_shot.ipynb) |

Los notebooks van en orden: el 02 usa el mismo preprocesamiento del 01, y el 03 usa el modelo que guarda el 02.

## Estructura

```
notebooks/   notebooks de Google Colab
scripts/     preprocesamiento_facial.py, con las funciones que comparten los tres notebooks
results/     figuras y métricas generadas (sin fotografías originales)
docs/        formato de autorización de uso de imagen y guía de ejecución
```

Los datos los mantengo en Google Drive (`MyDrive/Redes_Neuronales_V/`) y **no** están publicados aquí:

```
dataset_raw/<persona>/        fotos originales autorizadas
dataset_procesado/<persona>/  rostros alineados de 160×160 (los genera el Notebook 01)
fotos_demo/                   fotos nuevas para la demostración del Notebook 03
cache/  modelos/  results/
```

## Cómo ejecutarlo

1. Crear en Google Drive la carpeta `Redes_Neuronales_V/dataset_raw/` con una subcarpeta por persona.
2. Abrir cada notebook con el botón de Colab y activar la GPU (*Entorno de ejecución → Cambiar tipo de entorno*).
3. Ejecutar *Entorno de ejecución → Ejecutar todo* en el orden 01, 02 y 03.

En `docs/guia_ejecucion.md` dejé el paso a paso completo.

## Ética y cuidado de los datos

- Solo usé fotografías mías y de una persona adulta con autorización escrita para uso académico
  (`docs/autorizacion_uso_imagen.md`). No incluí menores de edad.
- En las fotos donde aparecían otras personas, recorté la imagen y difuminé a quienes no autorizaron su uso.
- Las fotografías originales y los rostros procesados están excluidos del repositorio en el `.gitignore`.
- En la demostración, el sistema difumina automáticamente cualquier rostro que no reconoce.

## Referencias

- Hermans, A., Beyer, L., y Leibe, B. (2017). In defense of the triplet loss for person re-identification. *arXiv:1703.07737*.
- Huang, G. B., Ramesh, M., Berg, T., y Learned-Miller, E. (2007). *Labeled Faces in the Wild: A database for studying face recognition in unconstrained environments* (Technical Report 07-49). University of Massachusetts, Amherst.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., y Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
- Schroff, F., Kalenichenko, D., y Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR*.
- Zhang, K., Zhang, Z., Li, Z., y Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. *IEEE Signal Processing Letters, 23*(10).
