import numpy as np
import tensorflow as tf

def test_single_ecg():
    # Modelin yolu
    model_path = "models/ecg_model.tflite"
    
    # TFLite Modelini Yükle
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Modelin Girdi (Input) ve Çıktı (Output) Detaylarını Al
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test için tek bir "Kritik (Anormal)" EKG Verisi oluşturalım.
    # Gerçek hayatta buraya hastadan gelen dizi verisi konulacak.
    # 750 örnek, 1 kanal
    print("Olası bir hastadan gelen EKG verisi simüle ediliyor...")
    test_data = np.random.randn(1, 750, 1).astype(np.float32)
    
    # Riskli bir durumu simüle etmek için ST yüksekliği ekliyoruz
    test_data[0, 400:450, 0] += 1.5 
    
    # Veriyi modele ver
    interpreter.set_tensor(input_details[0]['index'], test_data)

    # Tahmin işlemini çalıştır
    print("Yapay Zeka (CNN Modeli) durumu analiz ediyor...")
    interpreter.invoke()

    # Sonucu al
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    print("\n--- ANALİZ SONUCU ---")
    print(f"Risk Skoru (0.0 Normal - 1.0 Kritik): {prediction:.4f}")
    
    if prediction > 0.5:
        print("DURUM: DIKKAT! Anormal EKG kalibi (orn. STEMI) tespit edildi. Acil mudahale gerekebilir!")
    else:
        print("DURUM: Normal EKG kalibi.")

if __name__ == "__main__":
    test_single_ecg()
