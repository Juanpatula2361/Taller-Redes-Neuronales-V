"""
preprocesamiento_facial.py
==========================
Funciones de preprocesamiento de imágenes faciales usadas en los notebooks de la actividad
R2-A1-S10 "Redes Neuronales V" (Deep Learning - Conceptos, Universidad de Cundinamarca).

Flujo de preprocesamiento (el mismo para LFW y para el dataset propio):
    1. Carga robusta de la imagen (JPG, PNG, HEIC de iPhone, DNG/RAW) y corrección de la
       orientación EXIF, para que las fotos del celular no queden giradas.
    2. Detección del rostro y de cinco puntos faciales (ojos, nariz, comisuras) con MTCNN.
    3. Alineación: rotación de la imagen para que la línea de los ojos quede horizontal.
    4. Recorte cuadrado centrado en el rostro, con margen, y redimensionamiento a 160 x 160 px.
    5. Control de calidad: nitidez (varianza del laplaciano), brillo y contraste.

Autor: Juan Pablo Cristancho Tula (2026).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTENSIONES = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".dng", ".webp", ".bmp")

# Umbrales del control de calidad (para recortes de 160 x 160 px)
NITIDEZ_MIN = 25.0      # varianza del laplaciano; por debajo, la imagen se considera borrosa
BRILLO_MIN = 40.0       # media de gris (0-255)
BRILLO_MAX = 220.0
TAM_ROSTRO_MIN = 60     # lado mínimo del rostro detectado en la imagen original (px)


# --------------------------------------------------------------------------------------
# 1. Carga de imágenes
# --------------------------------------------------------------------------------------
def cargar_imagen(ruta: str, lado_max: int = 1600) -> np.ndarray:
    """Lee una imagen en RGB (uint8), corrige su orientación EXIF y limita su lado mayor.

    Soporta HEIC/HEIF (fotos de iPhone, requiere ``pillow-heif``) y DNG (RAW, requiere
    ``rawpy``). Reducir la resolución acelera la detección sin afectar el recorte final.
    """
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".dng":
        import rawpy  # importación diferida: solo si hay archivos RAW

        with rawpy.imread(ruta) as raw:
            img = Image.fromarray(raw.postprocess(use_camera_wb=True))
    else:
        if ext in (".heic", ".heif"):
            import pillow_heif

            pillow_heif.register_heif_opener()
        img = ImageOps.exif_transpose(Image.open(ruta))
    img = img.convert("RGB")
    img.thumbnail((lado_max, lado_max), Image.LANCZOS)
    return np.asarray(img).copy()


def listar_imagenes(carpeta: str) -> list[str]:
    """Rutas de las imágenes de una carpeta (no recursivo), en orden alfabético."""
    return sorted(
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.lower().endswith(EXTENSIONES) and not f.startswith(".")
    )


# --------------------------------------------------------------------------------------
# 2. Detección de rostros
# --------------------------------------------------------------------------------------
def crear_detector(dispositivo: str | None = None):
    """Crea el detector MTCNN (Multi-task Cascaded Convolutional Networks).

    Por defecto usa la GPU si TensorFlow la detecta (en Colab: Entorno de ejecución -> GPU).
    """
    from mtcnn import MTCNN

    if dispositivo is None:
        import tensorflow as tf

        dispositivo = "GPU:0" if tf.config.list_physical_devices("GPU") else "CPU:0"
    return MTCNN(device=dispositivo)


def detectar_rostros(img: np.ndarray, detector, confianza_min: float = 0.90,
                     tam_min: int = 30) -> list[dict]:
    """Devuelve los rostros detectados con confianza suficiente.

    Cada rostro es un diccionario con ``box`` = [x, y, ancho, alto], ``confidence`` y
    ``keypoints`` (left_eye, right_eye, nose, mouth_left, mouth_right).
    """
    caras = detector.detect_faces(img)
    salida = []
    for c in caras:
        x, y, w, h = [int(v) for v in c["box"]]
        if c["confidence"] >= confianza_min and min(w, h) >= tam_min:
            c = dict(c)
            c["box"] = [max(0, x), max(0, y), w, h]
            salida.append(c)
    return salida


def seleccionar_rostro(caras: list[dict], forma, criterio: str = "mayor"):
    """Elige un rostro: el de mayor área ("mayor") o el más cercano al centro ("centro")."""
    if not caras:
        return None
    if criterio == "centro":
        alto, ancho = forma[:2]
        cx, cy = ancho / 2, alto / 2
        return min(caras, key=lambda c: (c["box"][0] + c["box"][2] / 2 - cx) ** 2
                   + (c["box"][1] + c["box"][3] / 2 - cy) ** 2)
    return max(caras, key=lambda c: c["box"][2] * c["box"][3])


# --------------------------------------------------------------------------------------
# 3 y 4. Alineación y recorte
# --------------------------------------------------------------------------------------
def alinear_y_recortar(img: np.ndarray, cara: dict, tam: int = 160,
                       margen: float = 0.25) -> tuple[np.ndarray, float]:
    """Rota la imagen para nivelar los ojos y recorta un cuadrado centrado en el rostro.

    El ángulo se calcula con la recta que une ambos ojos: ``atan2(dy, dx)``. La rotación se
    hace alrededor del punto medio de los ojos y el recorte se centra en el rostro rotado.
    Si el cuadrado se sale de la imagen, se rellena por reflexión de los bordes.
    """
    ojo_i = np.asarray(cara["keypoints"]["left_eye"], dtype=float)
    ojo_d = np.asarray(cara["keypoints"]["right_eye"], dtype=float)
    dx, dy = ojo_d - ojo_i
    angulo = math.degrees(math.atan2(dy, dx))
    centro_ojos = tuple(((ojo_i + ojo_d) / 2).tolist())

    M = cv2.getRotationMatrix2D(centro_ojos, angulo, 1.0)
    rotada = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    x, y, w, h = cara["box"]
    cx, cy = M @ np.array([x + w / 2, y + h / 2, 1.0])
    lado = int(round(max(w, h) * (1 + 2 * margen)))
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    x1, y1 = x0 + lado, y0 + lado

    relleno = max(0, -x0, -y0, x1 - rotada.shape[1], y1 - rotada.shape[0])
    if relleno:
        rotada = cv2.copyMakeBorder(rotada, relleno, relleno, relleno, relleno, cv2.BORDER_REFLECT)
        x0, y0, x1, y1 = x0 + relleno, y0 + relleno, x1 + relleno, y1 + relleno

    recorte = cv2.resize(rotada[y0:y1, x0:x1], (tam, tam), interpolation=cv2.INTER_AREA)
    return recorte, angulo


def recorte_central(img: np.ndarray, tam: int = 160, fraccion: float = 0.55) -> np.ndarray:
    """Alternativa sin detector: recorte cuadrado central (útil para LFW si MTCNN falla)."""
    alto, ancho = img.shape[:2]
    lado = int(min(alto, ancho) * fraccion)
    y0, x0 = (alto - lado) // 2, (ancho - lado) // 2
    return cv2.resize(img[y0:y0 + lado, x0:x0 + lado], (tam, tam), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------------------
# 5. Control de calidad
# --------------------------------------------------------------------------------------
def metricas_calidad(recorte: np.ndarray) -> dict:
    """Nitidez (varianza del laplaciano), brillo y contraste del recorte en escala de grises."""
    gris = cv2.cvtColor(recorte, cv2.COLOR_RGB2GRAY)
    return {
        "nitidez": float(cv2.Laplacian(gris, cv2.CV_64F).var()),
        "brillo": float(gris.mean()),
        "contraste": float(gris.std()),
    }


def evaluar_calidad(calidad: dict, tam_rostro_px: int) -> list[str]:
    """Lista de advertencias de calidad (vacía si la imagen es aceptable)."""
    avisos = []
    if calidad["nitidez"] < NITIDEZ_MIN:
        avisos.append("borrosa")
    if calidad["brillo"] < BRILLO_MIN:
        avisos.append("oscura")
    if calidad["brillo"] > BRILLO_MAX:
        avisos.append("sobreexpuesta")
    if tam_rostro_px < TAM_ROSTRO_MIN:
        avisos.append("rostro pequeño")
    return avisos


def ecualizar_clahe(recorte: np.ndarray, limite: float = 2.0) -> np.ndarray:
    """Ecualización adaptativa (CLAHE) sobre la luminancia; se usa solo para análisis."""
    lab = cv2.cvtColor(recorte, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=limite, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# --------------------------------------------------------------------------------------
# Procesamiento completo
# --------------------------------------------------------------------------------------
@dataclass
class ResultadoRostro:
    recorte: np.ndarray
    box: list
    confianza: float
    angulo: float
    n_rostros: int
    calidad: dict = field(default_factory=dict)
    avisos: list = field(default_factory=list)
    escala: float = 1.0   # factor de escala con el que se logró la detección


def escalar(img: np.ndarray, factor: float) -> np.ndarray:
    """Redimensiona la imagen por un factor (INTER_AREA al reducir, INTER_CUBIC al ampliar)."""
    if factor == 1.0:
        return img
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def procesar_imagen(img: np.ndarray, detector, tam: int = 160, margen: float = 0.25,
                    criterio: str = "mayor", confianza_min: float = 0.90,
                    escalas: tuple = (1.0, 0.5, 1.5)):
    """Detecta, alinea, recorta y evalúa el rostro principal de una imagen RGB.

    La detección es multiescala: si MTCNN no encuentra un rostro en la resolución original,
    se reintenta con la imagen reducida y ampliada (el detector es sensible al tamaño
    relativo del rostro, por ejemplo con gafas deportivas o capucha).
    Devuelve un ``ResultadoRostro`` o ``None`` si no se detecta ningún rostro.
    """
    # Si la foto tiene una dominante de color fuerte (p. ej. luz verde o roja), MTCNN puede fallar;
    # en ese caso se detecta sobre una copia en escala de grises, pero se recorta la imagen a color.
    gris = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    intentos = [(img, f) for f in escalas] + [(gris, f) for f in escalas]
    for fuente, factor in intentos:
        img_det = escalar(fuente, factor)
        caras = detectar_rostros(img_det, detector, confianza_min=confianza_min)
        cara = seleccionar_rostro(caras, img_det.shape, criterio)
        if cara is not None:
            break
    else:
        return None
    img_e = escalar(img, factor)
    recorte, angulo = alinear_y_recortar(img_e, cara, tam=tam, margen=margen)
    calidad = metricas_calidad(recorte)
    tam_rostro = int(min(cara["box"][2], cara["box"][3]) / factor)
    avisos = evaluar_calidad(calidad, tam_rostro)
    box_original = [int(v / factor) for v in cara["box"]]
    return ResultadoRostro(recorte, box_original, float(cara["confidence"]), angulo,
                           len(caras), calidad, avisos, factor)


def procesar_todos_los_rostros(img: np.ndarray, detector, tam: int = 160,
                               margen: float = 0.25, confianza_min: float = 0.90):
    """Procesa todos los rostros de una foto (se usa en la demostración de reconocimiento)."""
    salida = []
    for cara in detectar_rostros(img, detector, confianza_min=confianza_min):
        recorte, angulo = alinear_y_recortar(img, cara, tam=tam, margen=margen)
        salida.append(ResultadoRostro(recorte, cara["box"], float(cara["confidence"]), angulo, 1))
    return salida


def difuminar_region(img: np.ndarray, box, fuerza: int = 51) -> np.ndarray:
    """Difumina (desenfoque gaussiano) una región; se usa para proteger rostros no autorizados."""
    x, y, w, h = [int(v) for v in box]
    salida = img.copy()
    region = salida[y:y + h, x:x + w]
    if region.size:
        k = max(3, (min(w, h) // 3) | 1)
        salida[y:y + h, x:x + w] = cv2.GaussianBlur(region, (max(k, fuerza | 1), max(k, fuerza | 1)), 0)
    return salida


def cargar_dataset_procesado(carpeta: str):
    """Carga ``carpeta/<persona>/*.png`` y devuelve (X uint8, etiquetas, nombres, rutas)."""
    nombres = sorted(d for d in os.listdir(carpeta)
                     if os.path.isdir(os.path.join(carpeta, d)) and not d.startswith("."))
    X, y, rutas = [], [], []
    for i, nombre in enumerate(nombres):
        for ruta in listar_imagenes(os.path.join(carpeta, nombre)):
            X.append(np.asarray(Image.open(ruta).convert("RGB")))
            y.append(i)
            rutas.append(ruta)
    return np.stack(X), np.asarray(y), nombres, rutas
