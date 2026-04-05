import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import os
import json

def build_ecg_cnn_model(input_shape=(750, 1)):
    """
    Basit bir 1D CNN Modeli (EKG Sinyalleri için)
    Girdi: (Örneklem Sayısı, Kanal Sayısı) -> Örn: 3 saniyelik 250Hz sinyal = 750 örnek
    Çıktı: İkili Sınıflandırma (0: Normal, 1: Anomali/STEMI)
    """
    model = Sequential([
        # 1. Konvolüsyon Bloğu
        Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        # 2. Konvolüsyon Bloğu
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # 3. Konvolüsyon Bloğu
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Tam Bağlı (Dense) Katmanlar
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Aşırı öğrenmeyi (Overfitting) engellemek için
        Dense(1, activation='sigmoid') # İkili sınıflandırma (0-1 arası olasılık)
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def create_dummy_dataset(num_samples=1000, input_length=750):
    """
    Eğitim için rastgele sentetik EKG verisi üretir.
    Gerçek senaryoda burası MIT-BIH veri setinden beslenecektir.
    """
    print(f"{num_samples} adet sentetik eğitim verisi oluşturuluyor...")
    # Rastgele sinyaller (Gerçek bir EKG'yi tam yansıtmaz, test amaçlıdır)
    X = np.random.randn(num_samples, input_length, 1)
    
    # Etiketler: Yarı yarıya normal(0) ve anormal(1)
    y = np.random.randint(0, 2, size=(num_samples, 1))
    
    # Eğer y=1 ise (anormal), sinyalin belli bir kısmına suni bir yükselti(ST elevation) ekle
    for i in range(num_samples):
        if y[i] == 1:
            X[i, 400:450, 0] += np.random.uniform(0.5, 1.5) # ST Yükselmesi efekti
            
    return X, y

def train_and_export_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Parametreler
    input_length = 750 # 3 saniye * 250Hz
    
    # Veriyi hazırla
    X_train, y_train = create_dummy_dataset(num_samples=2000, input_length=input_length)
    X_val, y_val = create_dummy_dataset(num_samples=400, input_length=input_length)
    
    # Modeli oluştur
    print("CNN Modeli oluşturuluyor...")
    model = build_ecg_cnn_model(input_shape=(input_length, 1))
    model.summary()
    
    # Eğit
    print("Model eğitimi başlıyor...")
    history = model.fit(X_train, y_train, 
                        epochs=5, 
                        batch_size=32, 
                        validation_data=(X_val, y_val),
                        verbose=1)
    
    # Keras formatında kaydet
    keras_model_path = os.path.join(model_dir, "ecg_cnn_model.h5")
    model.save(keras_model_path)
    print(f"Keras modeli şuraya kaydedildi: {keras_model_path}")
    
    # Mobile (Edge) cihazlar için TensorFlow Lite formatına çevir
    print("Model TFLite formatına dönüştürülüyor...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_model_path = os.path.join(model_dir, "ecg_model.tflite")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite modeli şuraya kaydedildi: {tflite_model_path}")
    
    # Eğitim geçmişini kaydet (opsiyonel)
    history_dict = history.history
    # numpy array'leri json'a çevrilebilir hale getirme
    for k in history_dict:
        history_dict[k] = [float(val) for val in history_dict[k]]
        
    with open(os.path.join(model_dir, "training_history.json"), 'w') as f:
        json.dump(history_dict, f)
        
    return tflite_model_path

if __name__ == "__main__":
    train_and_export_model()
