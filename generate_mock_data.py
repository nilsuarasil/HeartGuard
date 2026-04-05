import numpy as np
import pandas as pd
import time
import math

def generate_ecg_signal(duration_sec=30, sampling_rate=250, heart_rate=75, is_anomaly=False):
    """
    Basit bir sentetik EKG sinyali üretir.
    duration_sec: Sinyalin saniye cinsinden uzunluğu.
    sampling_rate: Saniyedeki örneklem sayısı (Hz).
    heart_rate: Dakikadaki atım sayısı (BPM).
    is_anomaly: True ise STEMI benzeri bir ST yükselmesi ekler.
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sampling_rate))
    
    # Kalp atımı frekansı
    beat_frequency = heart_rate / 60.0
    
    # Temel EKG bileşenleri (P, Q, R, S, T dalgaları için basitleştirilmiş sinyal)
    # R dalgası (Sharp peak)
    r_wave = np.sin(2 * np.pi * beat_frequency * t) ** 40 
    
    # P dalgası (Küçük, R'den önce)
    p_wave = 0.1 * np.sin(2 * np.pi * beat_frequency * (t + 0.15)) ** 10
    
    # T dalgası (R'den sonra)
    t_wave = 0.2 * np.sin(2 * np.pi * beat_frequency * (t - 0.25)) ** 10
    
    # Temel sinyal kombinasyonu
    ecg_signal = r_wave + p_wave + t_wave
    
    if is_anomaly:
        # STEMI Anomaly Simulation: ST segment elevation
        # T wave'den hemen önceki segmenti yükseltiriz
        st_elevation = 0.3 * np.sin(2 * np.pi * beat_frequency * (t - 0.1)) ** 8
        ecg_signal += st_elevation
        
    # Gürültü ekleme (Baseline wander ve yüksek frekanslı minik gürültü)
    noise = 0.05 * np.random.randn(len(t))
    baseline_wander = 0.1 * np.sin(2 * np.pi * 0.1 * t)
    
    ecg_signal = ecg_signal + noise + baseline_wander
    
    return t, ecg_signal

def generate_vitals(is_critical=False):
    """
    Nabız ve tansiyon (Sistolik/Diastolik) verisi üretir.
    """
    if not is_critical:
        hr = np.random.randint(60, 100)
        sys_bp = np.random.randint(110, 130)
        dia_bp = np.random.randint(70, 85)
    else:
        # Acil durum tetikleyecek veriler (örn: Çok yüksek nabız veya tansiyon)
        hr = np.random.choice([np.random.randint(40, 50), np.random.randint(120, 160)])
        sys_bp = np.random.randint(160, 200)
        dia_bp = np.random.randint(90, 120)
        
    return hr, sys_bp, dia_bp

if __name__ == "__main__":
    print("--- Normal Veri Üretimi ---")
    hr, sys, dia = generate_vitals(is_critical=False)
    print(f"Nabız: {hr} BPM, Tansiyon: {sys}/{dia} mmHg")
    t_norm, ecg_norm = generate_ecg_signal(duration_sec=3, is_anomaly=False)
    print(f"Normal EKG Sinyali Üretildi: {len(ecg_norm)} örnek.")
    
    print("\n--- Kritik (Acil Durum) Veri Üretimi ---")
    hr_crit, sys_crit, dia_crit = generate_vitals(is_critical=True)
    print(f"Nabız: {hr_crit} BPM, Tansiyon: {sys_crit}/{dia_crit} mmHg")
    t_anom, ecg_anom = generate_ecg_signal(duration_sec=3, is_anomaly=True)
    print(f"Anormal EKG Sinyali Üretildi: {len(ecg_anom)} örnek.")
    
    # Verileri CSV olarak kaydetme opsiyonu eklenecektir
    # pd.DataFrame({'Time': t_anom, 'ECG': ecg_anom}).to_csv('mock_abnormal_ecg.csv', index=False)
