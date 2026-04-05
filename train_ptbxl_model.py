"""
PTB-XL Gerçek Klinik Veri Seti ile CNN Modelinin Yeniden Eğitimi
-----------------------------------------------------------------
PTB-XL: 21.837 hastadan alınan 10 saniyelik 12-kanal EKG kayıtları.
Etiket: NORM (normal) veya MI (Miyokard Enfarktüsü / STEMI / NSTEMI)

Veri Kaynağı: https://physionet.org/content/ptb-xl/1.0.3/
"""

import os
import json
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ─── Ayarlar ───────────────────────────────────────────────────────────────────
PTBXL_PN_DIR   = "ptb-xl"          # PhysioNet kısa adı (wfdb.rdrecord için)
SAMPLE_RATE    = 100               # PTB-XL 100Hz veya 500Hz, 100Hz daha hafif
SIGNAL_SECONDS = 7                 # İlk 7 saniye (700 örnek @ 100Hz)
N_SAMPLES      = 700               # Model girdi uzunluğu
N_NORM         = 300               # İndirilecek Normal kayıt sayısı
N_MI           = 300               # İndirilecek MI kayıt sayısı
MODEL_DIR      = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def download_ptbxl_metadata():
    """PTB-XL metadata CSV'sini PhysioNet'ten indirir."""
    meta_path = "ptbxl_database.csv"
    if os.path.exists(meta_path):
        print("Metadata zaten mevcut, tekrar indirilmiyor.")
        return pd.read_csv(meta_path, index_col="ecg_id")
    
    print("PTB-XL metadata CSV indiriliyor (küçük dosya, hızlı)...")
    wfdb.dl_files('ptb-xl', '.', ['ptbxl_database.csv'])
    df = pd.read_csv(meta_path, index_col='ecg_id')
    print(f"Toplam kayıt: {len(df)}")
    return df


def parse_scp_codes(row):
    """scp_codes sütununu parse edip MI veya NORM içeriyor mu kontrol eder."""
    try:
        codes = json.loads(row.replace("'", '"'))
        has_mi   = any(k in codes for k in ['MI', 'AMI', 'IMI', 'ASMI', 'ILMI', 'ALMI', 'INJAS', 'INJIN', 'INJIL', 'INJLA', 'WPW'])
        has_norm = 'NORM' in codes
        return 'MI' if has_mi else ('NORM' if has_norm else 'OTHER')
    except Exception:
        return 'OTHER'


def load_record(filename_lr):
    """Tek bir PTB-XL kaydını indirip Lead II sinyalini döndürür."""
    try:
        # filename_lr örn: "records100/00001_lr" şeklinde olabilir
        record_path = filename_lr.replace('.hea', '')
        # Strip leading slash if present
        record_path = record_path.lstrip('/')
        record = wfdb.rdrecord(record_path, sampto=N_SAMPLES, pn_dir=PTBXL_PN_DIR)
        sig = record.p_signal[:N_SAMPLES, 1].astype(np.float32)
        if len(sig) < N_SAMPLES:
            sig = np.pad(sig, (0, N_SAMPLES - len(sig)))
        return sig
    except Exception as e:
        return None


def build_model(input_shape=(N_SAMPLES, 1)):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 7, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(64, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # 0 = NORM, 1 = MI
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


def train_and_export():
    # 1) Metadata
    df = download_ptbxl_metadata()
    df['label'] = df['scp_codes'].apply(parse_scp_codes)

    norm_records = df[df['label'] == 'NORM'].head(N_NORM)
    mi_records   = df[df['label'] == 'MI'].head(N_MI)
    
    print(f"Normal kayıt: {len(norm_records)}, MI kayıt: {len(mi_records)}")

    # 2) Sinyalleri İndir
    X, y = [], []

    print("Normal kayıtlar indiriliyor...")
    for i, (ecg_id, row) in enumerate(norm_records.iterrows()):
        sig = load_record(row['filename_lr'])
        if sig is not None:
            X.append(sig)
            y.append(0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(norm_records)} normal tamamlandı")

    print("MI (STEMI/NSTEMI) kayıtlar indiriliyor...")
    for i, (ecg_id, row) in enumerate(mi_records.iterrows()):
        sig = load_record(row['filename_lr'])
        if sig is not None:
            X.append(sig)
            y.append(1)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(mi_records)} MI tamamlandı")

    X = np.array(X).reshape(-1, N_SAMPLES, 1)
    y = np.array(y)

    print(f"\nToplam veri: {len(X)} | Normal: {np.sum(y==0)} | MI: {np.sum(y==1)}")

    # 3) Train / Validation Ayrımı
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 4) Model Eğitimi
    print("\nModel eğitiliyor (gerçek klinik veri)...")
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')
    ]

    model.fit(X_train, y_train,
              epochs=30,
              batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=callbacks,
              verbose=1)

    # 5) Kaydet
    keras_path = os.path.join(MODEL_DIR, "ecg_ptbxl_model.h5")
    model.save(keras_path)
    print(f"\nKeras modeli kaydedildi: {keras_path}")

    # 6) TFLite Dönüşümü
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(MODEL_DIR, "ecg_ptbxl_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model kaydedildi: {tflite_path}")
    print(f"\nGrafik cizmek icin: python plot_ecg.py --model {tflite_path}")

    return tflite_path


if __name__ == "__main__":
    train_and_export()
