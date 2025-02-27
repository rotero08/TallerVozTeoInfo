# Analizador Espectral de Voz

Este proyecto es una aplicación de escritorio en Python para el análisis en tiempo real de señales de audio, enfocada en la representación espectral y la evaluación de la precisión en la detección de frecuencias. La interfaz gráfica está desarrollada con **PyQt5** y las visualizaciones se realizan utilizando **Matplotlib**. Además, incorpora funciones de prueba automática que generan un tono de prueba para evaluar la precisión de la representación en frecuencia y permiten ajustar parámetros como la resolución de la FFT y el solapamiento del espectrograma.

---

## Características

- **Visualización en Tiempo Real:**  
  - Forma de onda de la señal de audio.
  - Espectro de frecuencia.
  - Espectrograma con representación en decibelios (dB).

- **Panel de Control:**  
  - Selección del dispositivo de entrada de audio.
  - Ajuste de la frecuencia máxima a visualizar.
  - Controles de grabación: iniciar, detener y guardar audio.

- **Pruebas y Evaluación:**  
  - Ejecución automática de una prueba de frecuencia (tono de prueba).
  - Contador que muestra el tiempo transcurrido durante la prueba.
  - Sliders para ajustar la frecuencia de prueba (rango de 50 a 2000 Hz) y la duración de la prueba (1 a 15 segundos).
  - Visualización en tiempo real de la frecuencia detectada y el error comparado con la frecuencia esperada.
  - Al finalizar la prueba, se restablecen los gráficos a su estado inicial y se muestra un mensaje con los resultados (éxito o fallo).

---

## Requisitos

- Python 3.x
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro)
- [SoundDevice](https://python-sounddevice.readthedocs.io/)
- [SciPy](https://www.scipy.org/)

Puedes instalar las dependencias necesarias mediante pip:

```bash
pip install numpy matplotlib PyQt5 sounddevice scipy
```

---

## Uso

1. **Clonar o descargar el repositorio.**
2. **Ejecutar el script principal:**  
   Abre una terminal y ejecuta:
   ```bash
   python nombre_del_script.py
   ```
   (Reemplaza `nombre_del_script.py` por el nombre del archivo que contiene el código).

3. **Interfaz:**  
   - En el panel de control a la izquierda, selecciona el dispositivo de entrada, ajusta la frecuencia máxima y utiliza los botones para iniciar o detener la grabación y guardar el audio.
   - En el grupo "Pruebas y Evaluación", ajusta la frecuencia de prueba y el tiempo de prueba mediante los sliders. Al presionar el botón "Ejecutar Prueba Automática", se realizará la prueba durante el tiempo configurado, mostrándose en tiempo real la frecuencia detectada y el error. Al finalizar, se mostrará un mensaje con los resultados.
