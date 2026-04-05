import numpy as np
import tensorflow as tf
import wfdb
import os

def test_with_mitbih():
    model_path = "models/ecg_model.tflite"
    
    # Modelin yüklenmesi
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("MIT-BIH Arrhythmia Veritabanından (Kayıt: 100) gerçek hasta verisi indiriliyor...")
    
    # wfdb.rdrecord ile PhysioNet üzerinden doğrudan kayıt 100'ü indiriyoruz
    # Sadece ilk 3 saniyelik veriyi alalım (Örneklem frekansı 360Hz)
    # 3 x 360 = 1080 örnek.
    # Ancak bizim modelimiz 750 örnek (3 sn * 250Hz) ile eğitildi (Mock verilerde)
    # Gerçek dünya senaryosunda 'resampling' yapılmalıdır. 
    # Şimdilik modelin beklediği boyuta uyum sağlamak için sadece 750 örneği keseceğiz.
    
    # Eğer henüz indirmediyse internetten çeker ve kaydeder
    record = wfdb.rdrecord('100', sampto=1000, pn_dir='mitdb')
    
    # record.p_signal, sinyali içerir (Örn: [1000 örnek, 2 kanal])
    # MLII kanalı (Genellikle indeks 0) alalım.
    ecg_signal = record.p_signal[:, 0]
    
    # Modelin input shape'i (750, 1) olduğu için ilk 750 örneği alıyoruz
    test_data = ecg_signal[:750].reshape(1, 750, 1).astype(np.float32)
    
    # Model için tahmine başla
    interpreter.set_tensor(input_details[0]['index'], test_data)
    
    print("\nYapay Zeka (CNN Modeli) Gerçek Hasta (Kayıt 100 - Normal Ritim) verisini analiz ediyor...")
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    print("\n--- ANALİZ SONUCU ---")
    print(f"Risk Skoru (0.0 Normal - 1.0 Kritik): {prediction:.4f}")
    
    if prediction > 0.5:
        print("DURUM: DIKKAT! Anormal EKG kalibi (orn. STEMI) tespit edildi. Acil mudahale gerekebilir!")
    else:
        print("DURUM: Normal EKG kalibi.")

    
    # İkinci Hasta Örneği (Kayıt: 200 - Ventriküler Aritmilerin bol olduğu hasta)
    print("\n------------------------------------------------------")
    print("MIT-BIH Arrhythmia Veritabanından (Kayıt: 200) riskli hasta verisi indiriliyor...")
    record_risk = wfdb.rdrecord('200', sampto=1000, pn_dir='mitdb')
    ecg_signal_risk = record_risk.p_signal[:, 0]
    test_data_risk = ecg_signal_risk[:750].reshape(1, 750, 1).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_data_risk)
    
    print("Yapay Zeka (CNN Modeli) Gerçek Hasta (Kayıt 200 - Aritmili Ritim) verisini analiz ediyor...")
    interpreter.invoke()
    
    prediction_risk = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    print("\n--- ANALİZ SONUCU ---")
    print(f"Risk Skoru (0.0 Normal - 1.0 Kritik): {prediction_risk:.4f}")
    
    if prediction_risk > 0.5:
        print("DURUM: DIKKAT! Anormal EKG kalibi (orn. STEMI) tespit edildi. Acil mudahale gerekebilir!")
    else:
        print("DURUM: Normal EKG kalibi.")

if __name__ == "__main__":
    test_with_mitbih()
