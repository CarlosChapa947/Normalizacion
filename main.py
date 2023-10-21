import cv2
import numpy as np
import matplotlib.pyplot as plt
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def calculateHistogram(imagen):

    if len(imagen.shape) == 2:
        histograma = np.zeros(256, dtype=int)
        for pixel_valor in imagen.ravel():
            histograma[pixel_valor] += 1

        return histograma
    elif len(imagen.shape) == 3:
        canal_r = imagen[:, :, 2]
        canal_g = imagen[:, :, 1]
        canal_b = imagen[:, :, 0]

        # Calcular los histogramas de los canales de color
        histograma_r = np.zeros(256, dtype=int)
        histograma_g = np.zeros(256, dtype=int)
        histograma_b = np.zeros(256, dtype=int)

        for pixel_valor in canal_r.ravel():
            histograma_r[pixel_valor] += 1

        for pixel_valor in canal_g.ravel():
            histograma_g[pixel_valor] += 1

        for pixel_valor in canal_b.ravel():
            histograma_b[pixel_valor] += 1

        return histograma_r, histograma_g, histograma_b



def calcular_histograma_normalizado(imagen_path):
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    histograma = np.zeros(256, dtype=int)
    for pixel_valor in imagen.ravel():
        histograma[pixel_valor] += 1

    total_pixeles = imagen.size
    histograma_normalizado = histograma / total_pixeles

    return histograma_normalizado

def binarizar_imagen(imagen_path, umbral):
    # Leer la imagen
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    #Gracias dios por np.where
    imagen_binaria = np.where(imagen >= umbral, 255, 0).astype(np.uint8)

    return imagen_binaria

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread("./Images/lena.png", cv2.IMREAD_GRAYSCALE)
    histo = calculateHistogram(img)
    plt.hist(histo)
    plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
