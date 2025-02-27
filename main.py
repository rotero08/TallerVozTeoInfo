import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sounddevice as sd
import scipy.fftpack as fftpack
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QLabel, QComboBox, QGroupBox, 
                            QSlider, QFileDialog, QProgressBar, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon
import threading
import time
import queue
import scipy.io.wavfile as wavfile
import os

class SoundWaveAnalyzer(QMainWindow):
    # Aplicación para análisis de voz en tiempo real con visualización espectral avanzada
    
    def __init__(self):
        super().__init__()
        
        # Configuración básica de la ventana
        self.setWindowTitle("Analizador Espectral de Voz")
        self.setGeometry(100, 100, 1100, 750)
        
        # Aplicar tema moderno
        self.apply_modern_theme()
        
        # Parámetros de captura y análisis de audio
        self.is_recording = False
        self.sample_rate = 44100     # Frecuencia de muestreo en Hz
        self.chunk_duration = 0.1    # Duración de cada fragmento en segundos
        self.max_display_freq = 5000 # Frecuencia máxima a mostrar en Hz
        self.analysis_window = 2     # Ventana de análisis en segundos
        self.fft_size = 2048         # Tamaño de ventana para FFT (aumentado para mejor resolución)
        self.overlap_ratio = 0.75    # Solapamiento para el espectrograma
        
        # Parámetros para el modo de prueba
        self.test_mode = False
        self.test_frequency = 440.0   # Frecuencia del tono de prueba en Hz (valor inicial)
        self.test_phase = 0.0         # Fase acumulada para la señal de prueba
        self.test_duration = 5000     # Duración de la prueba en milisegundos (valor inicial: 5 s)
        self.last_detected_freq = None
        self.last_error = None
        self.acceptable_error = 5.0   # Umbral en Hz para considerar la prueba exitosa
        
        # Buffers y variables de procesamiento
        self.audio_data = np.zeros(int(self.analysis_window * self.sample_rate))
        self.audio_queue = queue.Queue()
        self.start_time = 0
        
        # Inicializar interfaz de usuario
        self.setup_ui()
        
        # Temporizador para actualización en tiempo real
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(50)  # Actualización cada 50ms
        
    def apply_modern_theme(self):
        # Aplicación de tema moderno con gradientes y estilo neomórfico
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0a0e14;
                color: #e4e4e4;
            }
            QGroupBox {
                border: 2px solid #1e2530;
                border-radius: 12px;
                margin-top: 1.5em;
                padding: 15px;
                font-weight: bold;
                font-size: 14px;
                color: #8cc1ff;
                background-color: #14181f;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                background-color: #14181f;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #30475e, stop:1 #283747);
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: 12px;
                min-height: 20px;
                max-height: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a5875, stop:1 #2c3e50);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #283747, stop:1 #30475e);
            }
            QPushButton:disabled {
                background: #1e2530;
                color: #6d7680;
            }
            QComboBox {
                background-color: #242d39;
                border: 1px solid #3d4c5e;
                border-radius: 6px;
                padding: 4px 8px;
                color: #e4e4e4;
                selection-background-color: #3d5a80;
                min-height: 22px;
            }
            QComboBox::drop-down {
                border: 0px;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #242d39;
                border: 1px solid #3d4c5e;
                selection-background-color: #3d5a80;
            }
            QSlider {
                height: 20px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3d4c5e;
                height: 6px;
                background: #242d39;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5e87b0, stop:1 #4682b4);
                border: 2px solid #5e87b0;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QLabel {
                color: #e4e4e4;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #3d4c5e;
                border-radius: 5px;
                text-align: center;
                background-color: #242d39;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4682b4, stop:1 #5da2d5);
                border-radius: 4px;
            }
        """)
        
    def setup_ui(self):
        # Configuración de todos los elementos de la interfaz de usuario
        
        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout principal: dos columnas (izquierda: controles, derecha: visualizaciones)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_widget.setLayout(main_layout)
        
        # Panel de control (izquierda)
        control_panel = QGroupBox("Panel de Control")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(8)
        control_panel.setLayout(control_layout)
        
        # 1. Selector de dispositivo de audio
        device_group = QGroupBox("Dispositivo de Entrada")
        device_layout = QVBoxLayout()
        device_layout.setSpacing(5)
        device_group.setLayout(device_layout)
        
        self.device_selector = QComboBox()
        self.load_audio_devices()
        device_layout.addWidget(self.device_selector)
        control_layout.addWidget(device_group)
        
        # 2. Slider de frecuencia
        freq_group = QGroupBox("Configuración de Frecuencia")
        freq_layout = QVBoxLayout()
        freq_layout.setSpacing(5)
        freq_group.setLayout(freq_layout)
        
        self.freq_label = QLabel(f"Frecuencia máxima: {self.max_display_freq} Hz")
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(1000, 15000)
        self.freq_slider.setValue(self.max_display_freq)
        self.freq_slider.valueChanged.connect(self.update_max_frequency)
        freq_layout.addWidget(self.freq_label)
        freq_layout.addWidget(self.freq_slider)
        control_layout.addWidget(freq_group)
        
        # 3. Botones de grabación (uno debajo del otro)
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(8)
        
        self.record_btn = QPushButton("🎙️ Iniciar Grabación")
        self.record_btn.clicked.connect(self.start_recording)
        self.record_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2aa876, stop:1 #218c5d);
            color: white;
            padding: 5px 10px;
            min-height: 20px;
            max-height: 25px;
            font-size: 12px;
        """)
        buttons_layout.addWidget(self.record_btn)
        
        self.stop_btn = QPushButton("⏹️ Detener Grabación")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
            color: white;
            padding: 5px 10px;
            min-height: 20px;
            max-height: 25px;
            font-size: 12px;
        """)
        buttons_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("💾 Guardar Audio")
        self.save_btn.clicked.connect(self.save_audio_file)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
            color: white;
            padding: 5px 10px;
            min-height: 20px;
            max-height: 25px;
            font-size: 12px;
        """)
        buttons_layout.addWidget(self.save_btn)
        control_layout.addLayout(buttons_layout)
        
        # 4. Estado y nivel de audio
        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)
        self.status_label = QLabel("Estado: Listo para grabar")
        self.status_label.setStyleSheet("color: #8cc1ff; font-weight: bold; font-size: 13px;")
        self.level_label = QLabel("Nivel de Audio:")
        self.audio_level = QProgressBar()
        self.audio_level.setRange(0, 100)
        self.audio_level.setValue(0)
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.level_label)
        status_layout.addWidget(self.audio_level)
        control_layout.addLayout(status_layout)
        
        # 5. Pruebas y Evaluación
        test_group = QGroupBox("Pruebas y Evaluación")
        test_layout = QVBoxLayout()
        test_layout.setSpacing(5)
        test_group.setLayout(test_layout)
        
        # Botón para iniciar la prueba (se ejecuta automáticamente por un tiempo fijo)
        self.start_test_btn = QPushButton("Ejecutar Prueba Automática")
        self.start_test_btn.clicked.connect(self.start_test_tone)
        self.start_test_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2aa876, stop:1 #218c5d);
            color: white;
            padding: 5px 10px;
            font-size: 12px;
        """)
        test_layout.addWidget(self.start_test_btn)
        
        # Slider para ajustar la frecuencia de prueba
        test_freq_layout = QHBoxLayout()
        test_freq_label = QLabel("Frecuencia de prueba (Hz):")
        self.test_freq_value_label = QLabel("440")
        self.test_freq_slider = QSlider(Qt.Horizontal)
        self.test_freq_slider.setRange(50, 2000)
        self.test_freq_slider.setValue(int(self.test_frequency))
        self.test_freq_slider.valueChanged.connect(self.update_test_frequency)
        test_freq_layout.addWidget(test_freq_label)
        test_freq_layout.addWidget(self.test_freq_slider)
        test_freq_layout.addWidget(self.test_freq_value_label)
        test_layout.addLayout(test_freq_layout)
        
        # Slider para ajustar el tiempo de prueba (segundos)
        test_duration_layout = QHBoxLayout()
        test_duration_label = QLabel("Tiempo de prueba (s):")
        self.test_duration_value_label = QLabel("5")
        self.test_duration_slider = QSlider(Qt.Horizontal)
        self.test_duration_slider.setRange(1, 15)
        self.test_duration_slider.setValue(5)
        self.test_duration_slider.valueChanged.connect(self.update_test_duration)
        test_duration_layout.addWidget(test_duration_label)
        test_duration_layout.addWidget(self.test_duration_slider)
        test_duration_layout.addWidget(self.test_duration_value_label)
        test_layout.addLayout(test_duration_layout)
        
        # Etiqueta para mostrar el tiempo transcurrido durante la prueba
        self.test_timer_label = QLabel("Tiempo transcurrido: 0.0 s")
        test_layout.addWidget(self.test_timer_label)
        
        # Etiquetas para mostrar frecuencia detectada y error
        self.detected_freq_label = QLabel("Frecuencia Detectada: -- Hz")
        self.freq_error_label = QLabel("Error: -- Hz")
        test_layout.addWidget(self.detected_freq_label)
        test_layout.addWidget(self.freq_error_label)
        
        control_layout.addWidget(test_group)
        
        # Panel de visualizaciones (derecha)
        vis_panel = QGroupBox("Análisis de Señal")
        vis_layout = QVBoxLayout()
        vis_layout.setSpacing(8)
        vis_panel.setLayout(vis_layout)
        
        # Configuración del tema para matplotlib
        plt.style.use('dark_background')
        
        # Visualización de forma de onda
        self.waveform_fig = Figure(figsize=(8, 2), dpi=100, facecolor='#14181f')
        self.waveform_canvas = FigureCanvas(self.waveform_fig)
        self.waveform_ax = self.waveform_fig.add_subplot(111)
        self.waveform_ax.set_title("Forma de Onda", fontsize=10, color='#8cc1ff')
        self.waveform_ax.set_xlabel("Tiempo (s)", fontsize=9, color='#e4e4e4')
        self.waveform_ax.set_ylabel("Amplitud", fontsize=9, color='#e4e4e4')
        self.waveform_ax.tick_params(colors='#e4e4e4', labelsize=8)
        self.waveform_ax.set_facecolor('#0a0e14')
        self.waveform_line, = self.waveform_ax.plot(
            np.linspace(0, self.analysis_window, len(self.audio_data)), 
            self.audio_data, '-', lw=1, color='#5da2d5'
        )
        self.waveform_fig.tight_layout(pad=1.0)
        vis_layout.addWidget(self.waveform_canvas)
        
        # Visualización del espectro de frecuencia
        self.spectrum_fig = Figure(figsize=(8, 2), dpi=100, facecolor='#14181f')
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)
        self.spectrum_ax.set_title("Espectro de Frecuencia", fontsize=10, color='#8cc1ff')
        self.spectrum_ax.set_xlabel("Frecuencia (Hz)", fontsize=9, color='#e4e4e4')
        self.spectrum_ax.set_ylabel("Magnitud", fontsize=9, color='#e4e4e4')
        self.spectrum_ax.tick_params(colors='#e4e4e4', labelsize=8)
        self.spectrum_ax.set_facecolor('#0a0e14')
        self.spectrum_line, = self.spectrum_ax.plot([], [], '-', lw=1, color='#ff7f7f')
        self.spectrum_fig.tight_layout(pad=1.0)
        vis_layout.addWidget(self.spectrum_canvas)
        
        # Visualización del espectrograma
        self.spectrogram_fig = Figure(figsize=(8, 3), dpi=100, facecolor='#14181f')
        self.spectrogram_canvas = FigureCanvas(self.spectrogram_fig)
        self.spectrogram_ax = self.spectrogram_fig.add_subplot(111)
        self.spectrogram_ax.set_title("Espectrograma", fontsize=10, color='#8cc1ff')
        self.spectrogram_ax.set_xlabel("Tiempo (s)", fontsize=9, color='#e4e4e4')
        self.spectrogram_ax.set_ylabel("Frecuencia (Hz)", fontsize=9, color='#e4e4e4')
        self.spectrogram_ax.tick_params(colors='#e4e4e4', labelsize=8)
        self.spectrogram_ax.set_facecolor('#0a0e14')
        
        empty_spectrogram = np.zeros((100, 100))
        self.spectrogram_img = self.spectrogram_ax.imshow(
            empty_spectrogram, 
            aspect='auto', 
            origin='lower', 
            cmap='plasma',
            extent=[0, self.analysis_window, 0, self.max_display_freq],
            vmin=-80,
            vmax=0
        )
        self.colorbar = self.spectrogram_fig.colorbar(self.spectrogram_img)
        self.colorbar.ax.tick_params(colors='#e4e4e4', labelsize=8)
        self.colorbar.set_label('Intensidad (dB)', color='#e4e4e4', fontsize=9)
        self.spectrogram_fig.tight_layout(pad=1.0)
        vis_layout.addWidget(self.spectrogram_canvas)
        
        # Agregar paneles al layout principal
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(vis_panel, 4)
        
    def load_audio_devices(self):
        # Carga los dispositivos de entrada de audio disponibles
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                self.device_selector.addItem(f"{device['name']} (ID: {i})", i)
                
    def audio_callback(self, indata, frames, time_info, status):
        # Callback para el stream de audio
        if status:
            print(f"Error de grabación: {status}")
        self.audio_queue.put(indata.copy())
        
    def process_audio_stream(self):
        # Procesamiento continuo del flujo de audio (modo micrófono)
        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=1)
                new_samples = len(chunk)
                self.audio_data = np.roll(self.audio_data, -new_samples)
                self.audio_data[-new_samples:] = chunk.flatten()
                if len(chunk) > 0:
                    level = min(100, int(np.sqrt(np.mean(chunk**2)) * 400))
                    self.audio_level.setValue(level)
            except queue.Empty:
                pass
            
    def process_test_tone_stream(self):
        # Procesamiento continuo del tono de prueba
        chunk_size = int(self.chunk_duration * self.sample_rate)
        t = np.arange(chunk_size) / self.sample_rate
        while self.test_mode:
            chunk = np.sin(2 * np.pi * self.test_frequency * (t + self.test_phase))
            self.test_phase += chunk_size / self.sample_rate
            new_samples = chunk_size
            self.audio_data = np.roll(self.audio_data, -new_samples)
            self.audio_data[-new_samples:] = chunk
            level = min(100, int(np.sqrt(np.mean(chunk**2)) * 400))
            self.audio_level.setValue(level)
            time.sleep(self.chunk_duration)
            
    def update_visualizations(self):
        if not (self.is_recording or self.test_mode):
            return
            
        # Actualizar forma de onda
        self.waveform_line.set_ydata(self.audio_data)
        y_max = max(0.1, np.max(np.abs(self.audio_data)))
        self.waveform_ax.set_ylim(-y_max, y_max)
        self.waveform_canvas.draw_idle()
        
        # Calcular espectro de frecuencia
        n_samples = len(self.audio_data)
        windowed_data = self.audio_data * np.hanning(n_samples)
        spectrum = np.abs(fftpack.fft(windowed_data))
        spectrum = spectrum[:n_samples//2]
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
        frequencies = np.linspace(0, self.sample_rate/2, len(spectrum))
        freq_mask = frequencies <= self.max_display_freq
        
        self.spectrum_line.set_data(frequencies[freq_mask], spectrum[freq_mask])
        self.spectrum_ax.set_xlim(0, self.max_display_freq)
        self.spectrum_ax.set_ylim(0, 1.1)
        self.spectrum_canvas.draw_idle()
        
        # Actualizar espectrograma mediante STFT
        step_size = int((1 - self.overlap_ratio) * self.fft_size)
        if step_size <= 0:
            step_size = 1
        if len(self.audio_data) >= self.fft_size:
            frame_indices = np.arange(0, len(self.audio_data) - self.fft_size + 1, step_size)
            if len(frame_indices) > 1:
                frames = np.zeros((len(frame_indices), self.fft_size))
                for i, start_idx in enumerate(frame_indices):
                    frames[i] = self.audio_data[start_idx:start_idx + self.fft_size] * np.hanning(self.fft_size)
                spectral_frames = np.abs(fftpack.fft(frames))[:, :self.fft_size//2]
                epsilon = 1e-10
                spectral_frames_db = 20 * np.log10(spectral_frames + epsilon)
                min_db, max_db = -100, 0
                spectral_frames_db = np.clip(spectral_frames_db, min_db, max_db)
                spectrogram_freqs = np.linspace(0, self.sample_rate/2, self.fft_size//2)
                freq_display_mask = spectrogram_freqs <= self.max_display_freq
                self.spectrogram_img.set_data(spectral_frames_db[:, freq_display_mask].T)
                self.spectrogram_img.set_extent([0, self.analysis_window, 0, self.max_display_freq])
                self.spectrogram_img.set_clim(min_db, max_db)
                self.spectrogram_canvas.draw_idle()
                
        # Si el modo de prueba está activado, detectar la frecuencia y almacenar los resultados
        if self.test_mode:
            windowed_data_full = self.audio_data * np.hanning(len(self.audio_data))
            spectrum_full = np.abs(fftpack.fft(windowed_data_full))
            spectrum_half = spectrum_full[:len(self.audio_data)//2]
            frequencies_full = np.linspace(0, self.sample_rate/2, len(spectrum_half))
            if len(spectrum_half) > 1:
                idx = np.argmax(spectrum_half[1:]) + 1
                detected_freq = frequencies_full[idx]
                error = abs(detected_freq - self.test_frequency)
                self.last_detected_freq = detected_freq
                self.last_error = error
                self.detected_freq_label.setText(f"Frecuencia Detectada: {detected_freq:.1f} Hz")
                self.freq_error_label.setText(f"Error: {error:.1f} Hz")
        
    def update_max_frequency(self):
        self.max_display_freq = self.freq_slider.value()
        self.freq_label.setText(f"Frecuencia máxima: {self.max_display_freq} Hz")
        self.spectrum_ax.set_xlim(0, self.max_display_freq)
        self.spectrogram_ax.set_ylim(0, self.max_display_freq)
        self.spectrogram_img.set_extent([0, self.analysis_window, 0, self.max_display_freq])
        
    def update_fft_size(self):
        # Actualizar tamaño de FFT basado en el valor del slider (2^valor)
        exponent = self.fft_slider.value()
        self.fft_size = 2 ** exponent
        self.fft_size_label.setText(f"{self.fft_size} (2^{exponent})")
        
    def update_overlap_ratio(self):
        # Actualizar solapamiento basado en el valor del slider (porcentaje)
        overlap_percent = self.overlap_slider.value()
        self.overlap_ratio = overlap_percent / 100.0
        self.overlap_label.setText(f"{overlap_percent}%")
        
    def update_test_frequency(self):
        # Actualizar la frecuencia de prueba y mostrarla en la etiqueta
        self.test_frequency = self.test_freq_slider.value()
        self.test_freq_value_label.setText(str(self.test_frequency))
        
    def update_test_duration(self):
        # Actualizar el tiempo de prueba mostrado (en segundos)
        duration = self.test_duration_slider.value()
        self.test_duration_value_label.setText(str(duration))
        
    def update_test_timer(self):
        # Actualizar el contador de tiempo transcurrido en la prueba
        elapsed = time.time() - self.test_start_time
        self.test_timer_label.setText(f"Tiempo transcurrido: {elapsed:.1f} s")
        
    def start_recording(self):
        if self.is_recording:
            return
        try:
            # Si se está en modo de prueba, detenerlo
            if self.test_mode:
                self.stop_test_tone()
            device_idx = self.device_selector.currentData()
            if device_idx is None:
                device_idx = sd.default.device[0]
            chunk_size = int(self.chunk_duration * self.sample_rate)
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=chunk_size,
                device=device_idx,
                channels=1,
                callback=self.audio_callback
            )
            self.is_recording = True
            self.audio_queue = queue.Queue()
            self.stream.start()
            self.processing_thread = threading.Thread(target=self.process_audio_stream)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.status_label.setText("Estado: Grabando...")
            self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 13px;")
            self.record_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.start_time = time.time()
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 13px;")
            
    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.status_label.setText("Estado: Grabación detenida")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 13px;")
        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
    def start_test_tone(self):
        # Iniciar modo de prueba de frecuencia (se ejecuta automáticamente por un tiempo fijo)
        if self.test_mode:
            return
        # Si se está grabando con micrófono, detenerlo
        if self.is_recording:
            self.stop_recording()
        self.test_mode = True
        self.test_phase = 0.0
        self.start_test_btn.setEnabled(False)
        # Actualizar parámetros de prueba según los sliders
        self.test_frequency = self.test_freq_slider.value()
        self.test_duration = self.test_duration_slider.value() * 1000  # convertir a ms
        self.test_start_time = time.time()
        # Iniciar el contador de tiempo
        self.test_elapsed_timer = QTimer()
        self.test_elapsed_timer.timeout.connect(self.update_test_timer)
        self.test_elapsed_timer.start(100)
        self.status_label.setText("Estado: Modo Prueba Activado")
        self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 13px;")
        self.test_thread = threading.Thread(target=self.process_test_tone_stream)
        self.test_thread.daemon = True
        self.test_thread.start()
        # Programar la finalización automática de la prueba
        QTimer.singleShot(self.test_duration, self.stop_test_tone)
        
    def stop_test_tone(self):
        # Detener modo de prueba de frecuencia y procesar resultados
        if not self.test_mode:
            return
        self.test_mode = False
        if hasattr(self, 'test_thread'):
            self.test_thread.join(timeout=1)
        if hasattr(self, 'test_elapsed_timer'):
            self.test_elapsed_timer.stop()
            self.test_timer_label.setText("Tiempo transcurrido: 0.0 s")
        self.start_test_btn.setEnabled(True)
        self.status_label.setText("Estado: Modo Prueba Finalizado")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 13px;")
        # Preparar mensaje con resultados
        if self.last_detected_freq is not None and self.last_error is not None:
            result = "EXITOSO" if self.last_error <= self.acceptable_error else "FALLIDO"
            message = (f"Prueba de Frecuencia:\n"
                       f"Frecuencia esperada: {self.test_frequency} Hz\n"
                       f"Frecuencia detectada: {self.last_detected_freq:.1f} Hz\n"
                       f"Error: {self.last_error:.1f} Hz\n"
                       f"Resultado: {result}\n\n"
                       "Se ajustó la resolución del espectrograma según los parámetros configurados.")
        else:
            message = "No se obtuvieron datos de la prueba."
        QMessageBox.information(self, "Resultado de la Prueba", message)
        self.reset_visualizations()
        
    def reset_visualizations(self):
        # Restablecer el buffer de audio y actualizar gráficos
        self.audio_data = np.zeros(int(self.analysis_window * self.sample_rate))
        self.waveform_line.set_ydata(self.audio_data)
        self.waveform_ax.set_ylim(-1, 1)
        self.waveform_canvas.draw_idle()
        # Limpiar espectro y espectrograma
        self.spectrum_line.set_data([], [])
        self.spectrum_ax.set_xlim(0, self.max_display_freq)
        self.spectrum_ax.set_ylim(0, 1.1)
        self.spectrum_canvas.draw_idle()
        empty_spectrogram = np.zeros((100, 100))
        self.spectrogram_img.set_data(empty_spectrogram)
        self.spectrogram_img.set_extent([0, self.analysis_window, 0, self.max_display_freq])
        self.spectrogram_canvas.draw_idle()
        
    def save_audio_file(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Guardar Archivo de Audio", "", "Archivos WAV (*.wav)"
            )
            if file_path:
                normalized_audio = self.audio_data / np.max(np.abs(self.audio_data))
                audio_int = (normalized_audio * 32767).astype(np.int16)
                wavfile.write(file_path, self.sample_rate, audio_int)
                self.status_label.setText(f"Audio guardado en: {os.path.basename(file_path)}")
                self.status_label.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 13px;")
        except Exception as e:
            self.status_label.setText(f"Error al guardar: {str(e)}")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 13px;")

def execute_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    main_window = SoundWaveAnalyzer()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    execute_app()
