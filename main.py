#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para treinamento de um modelo de reconhecimento de sinais (Libras)
utilizando MediaPipe para extração de pontos-chave e uma rede neural LSTM
para classificação de sequências de vídeo.
"""

# --- 1. Importação de Bibliotecas ---
import os
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# --- 2. Configuração Global ---

# Configurações do MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Configurações do Dataset e Pré-processamento
# IMPORTANTE: Certifique-se de que este caminho aponta para a pasta onde seus vídeos estão.
# A estrutura esperada é:
# - samples/
#   - obrigado/
#     - video1.mp4, video2.mp4, ...
#   - nada/
#     - videoA.mp4, videoB.mp4, ...
DATA_PATH = os.path.join('samples')
ACTIONS = np.array(['obrigado', 'nada'])
MAX_FRAMES = 30  # Número de frames por sequência de vídeo

# Configurações de Treinamento
TEST_SPLIT_SIZE = 0.2
VALIDATION_SPLIT_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 32
RANDOM_STATE = 42

# --- 3. Funções de Processamento de Dados ---

def mediapipe_detection(image: np.ndarray, model: mp_holistic.Holistic) -> Tuple[np.ndarray, Any]:
    """
    Processa uma imagem com o modelo MediaPipe Holistic para detectar pontos-chave.

    Args:
        image (np.ndarray): O frame do vídeo (em formato BGR).
        model (mp_holistic.Holistic): A instância do modelo MediaPipe Holistic.

    Returns:
        Tuple[np.ndarray, Any]: Uma tupla contendo a imagem original e os resultados da detecção.
    """
    # Converte a cor da imagem de BGR para RGB para processamento pelo MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    # Realiza a detecção
    results = model.process(image_rgb)
    
    # Retorna a imagem ao formato original
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    return image_bgr, results

def extract_keypoints(results: Any) -> np.ndarray:
    """
    Extrai e concatena os pontos-chave de pose, mão esquerda e mão direita dos resultados do MediaPipe.

    Args:
        results (Any): O objeto de resultados retornado pelo MediaPipe.

    Returns:
        np.ndarray: Um array NumPy achatado contendo todos os pontos-chave.
    """
    # Extrai pontos-chave da pose, preenchendo com zeros se não houver detecção
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
        
    # Extrai pontos-chave da mão esquerda
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
        
    # Extrai pontos-chave da mão direita
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
        
    return np.concatenate([pose, lh, rh])

def load_sequences_and_labels(data_path: str, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega os dados de vídeo, extrai pontos-chave e os organiza em sequências e rótulos.

    Args:
        data_path (str): O caminho para o diretório raiz do dataset.
        actions (np.ndarray): Um array com os nomes das classes (ações).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Uma tupla contendo o array de sequências (X) e o array de rótulos (y) em formato one-hot.
    """
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    print("Iniciando carregamento e processamento dos dados...")
    for action in actions:
        action_path = os.path.join(data_path, action)
        if not os.path.isdir(action_path):
            print(f"Aviso: Diretório não encontrado para a ação '{action}'. Pulando.")
            continue

        video_files = [f for f in os.listdir(action_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        print(f"Processando {len(video_files)} vídeos para a ação '{action}'...")

        for video_file in video_files:
            video_path = os.path.join(action_path, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Erro ao abrir o arquivo de vídeo: {video_path}")
                continue

            frames = []
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    frames.append(keypoints)
            cap.release()

            # Normaliza a sequência para ter um comprimento fixo (MAX_FRAMES)
            if frames:
                if len(frames) > MAX_FRAMES:
                    # Se o vídeo for longo, seleciona frames uniformemente espaçados
                    indices = np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)
                    frames = [frames[i] for i in indices]
                elif len(frames) < MAX_FRAMES:
                    # Se o vídeo for curto, preenche com zeros no final (padding)
                    padding = [np.zeros(frames[0].shape) for _ in range(MAX_FRAMES - len(frames))]
                    frames.extend(padding)
                
                sequences.append(frames)
                labels.append(label_map[action])

    print("\nResumo do Carregamento:")
    for action in actions:
        count = sum(1 for label in labels if label == label_map[action])
        print(f"  - Ação '{action}': {count} sequências carregadas.")
    
    return np.array(sequences), to_categorical(labels).astype(int)

# --- 4. Funções de Modelo e Treinamento ---

def build_lstm_model(input_shape: Tuple[int, int], num_classes: int) -> Sequential:
    """
    Constrói, compila e retorna o modelo LSTM.

    Args:
        input_shape (Tuple[int, int]): A forma dos dados de entrada (frames, features).
        num_classes (int): O número de classes de saída.

    Returns:
