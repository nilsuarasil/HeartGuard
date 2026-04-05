import numpy as np
import tensorflow as tf
import os
import sys

def test_from_txt(file_path):
    if not os.path.exists(file_path):
        print(f"HATA: '{file_path}' bulunamadı. Lütfen TXT dosyanızın burada olduğundan emin olun.")
        sys.exit(1)

    print(f"'{file_path}' dosyasından veriler okunuyor...")
    
    try:
        # Txt dosyasındaki verileri Oku.
        # Varsayım: Her satırda bir EKG voltaj (sayı) değeri var, örn: -0.15, 0.45, 1.2 vs.
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Temizle ve virgülle vs ayrılmışsa ayırarak float array'ine dönüştür
        data = []
        for line in lines:
            line = line.strip()
            if not line or "sure" in line.lower() or "genlik" in line.lower() or "time" in line.lower():
                continue # Başlık satırlarını veya boş satırları atla
            
            # Virgülü veya noktayı standartlaştır
            parts = line.replace(',', ' ').split()
            
            if len(parts) >= 2:
                # Format: 0.000, 0.062 (Zaman, Genlik)
                # İkinci sütun (Genlik/mV) değerini al
                val_str = parts[1]
            else:
                # Format: sadece genlik değeri varsa
                val_str = parts[0]
                
            try:
                data.append(float(val_str))
            except ValueError:
                pass # Metin vb varsa atla
                    
        signal = np.array(data)
        
    except Exception as e:
        print(f"Veri okunurken hata oluştu: {e}")
        sys.exit(1)
        
    print(f"{len(signal)} adet sayısal eşleşme bulundu.")
    
    if len(signal) == 0:
         print("HATA: Dosya içinde okunabilir geçerli bir EKG sayısal verisi bulunamadı.")
         sys.exit(1)

    # Bizim modelimiz 750 uzunluğunda veri bekliyor
    # Veriyi modele uygun boyuta getirelim (Pad (doldurma) veya Cut (kesme))
    target_length = 750
    if len(signal) > target_length:
        print(f"Uyarı: Girdiğiniz veri 750 örneklemden daha uzun. Yapay Zeka için ilk 750'si dikkate alınıyor.")
        signal = signal[:target_length]
    elif len(signal) < target_length:
         print(f"Uyarı: Girdiğiniz veri 750 örneklemden kısa. Yapay Zeka için sonu sıfırlarla tamamlanıyor.")
         pad_length = target_length - len(signal)
         signal = np.pad(signal, (0, pad_length), 'constant', constant_values=(0, 0))
         
    # Model shape'ine uyarla: (1, 750, 1)
    test_data = signal.reshape(1, 750, 1).astype(np.float32)

    # -- Model Yükleme ve Tahmin --
    model_path = "models/ecg_model.tflite"
    if not os.path.exists(model_path):
        print("HATA: Model dosyası bulunamadı. Önce train_cnn_model.py çalıştırılmalı.")
        sys.exit(1)
        
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], test_data)

    print("\nYapay Zeka (CNN Modeli) kendi txt verinizi analiz ediyor...")
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    print("\n--- ANALİZ SONUCU ---")
    print(f"Risk Skoru (0.000 Normal - 1.000 Kritik): {prediction:.4f}")

    if prediction > 0.5:
        print("DURUM: DIKKAT! Anormal EKG kalibi (orn. STEMI) tespit edildi. Acil mudahale gerekebilir!")
    else:
        print("DURUM: Normal EKG kalibi.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım Hatası.")
        print("Doğru Kullanım: python test_from_txt.py sizin_dosyaniz.txt")
    else:
        file_to_test = sys.argv[1]
        test_from_txt(file_to_test)
