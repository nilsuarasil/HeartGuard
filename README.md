# HeartGuard ❤️🛡️

HeartGuard, klinik EKG (Elektrokardiyografi) sinyallerini ve hayati bulguları derin öğrenme ve makine öğrenmesi modelleri kullanarak analiz eden, gerçek zamanlı bir **Risk Analiz ve Gösterge Paneli (Dashboard)** uygulamasıdır. Özellikle Miyokard Enfarktüsü (Kalp Krizi - STEMI/NSTEMI) ve ritim bozukluklarını (Aritmi) tespit etmek için tasarlanmıştır.

## 📊 Kullanılan Veri Tabanları (Eğitim Verileri)

Projenin yapay zeka modelleri ve canlı testleri, dünyaca kabul görmüş klinik tıp veri setlerine dayanmaktadır:

1. **PTB-XL Veri Tabanı:**
   - Derin öğrenme modelinin kalbi bu veri setine dayanır. 21.837 hastadan alınan 10 saniyelik 12 kanallı yüksek kaliteli klinik EKG kayıtlarını içerir.
   - Projede **1D CNN (Konvolüsyonel Sinir Ağı)** modeli, bu verileri kullanarak "Normal" ve "MI (Miyokard Enfarktüsü)" ayrımını yapmak üzere `train_ptbxl_model.py` içerisinde eğitilmiştir.
2. **MIT-BIH Aritmi Veri Tabanı:**
   - Dashboard (`main.py`) üzerinde gerçek zamanlı test ve simülasyonlar için kullanılır. Normal sinüs ritminden ventriküler fibrilasyona kadar farklı aritmi türlerini canlı olarak analiz etmek için sistem tarafından `wfdb` kütüphanesi ile veritabanından dinamik olarak çekilir.
3. **Sentetik Veri Üretimi:**
   - Test senaryoları için ST Segment Yükselmesi (STEMI) gibi spesifik kalp krizi bulgularını simüle eden yapay veri üretici algoritmalar.
4. **Vital Bulgular:**
   - Hastanın Nabız, Sistolik ve Diastolik Tansiyon değerlerini yaşamsal tablo özelinde değerlendirip sınıflandıran bir Rastgele Orman (Random Forest) modeli mevcuttur.

## ⚙️ Süreç: Sistem Nasıl İşliyor?

1. **Model Eğitimi (Derin Öğrenme):**
   - `train_ptbxl_model.py` ve `train_cnn_model.py` üzerinden EKG sinyalleri işlenir. 1D CNN katmanları kullanılarak zaman serisi EKG dalgalarından öznitelikler çıkarılır.
   - Eğitilen büyük boyutlu Keras (`.h5`) modelleri, masaüstü uygulamada tamamen çevrimdışı ve saniyeler içinde çalışabilmesi için optimize edilerek **TensorFlow Lite (`.tflite`)** formatına dönüştürülür.

2. **Gerçek Zamanlı Analiz (Dashboard):**
   - `main.py` çalıştırıldığında bir Tkinter masaüstü arayüzü açılır.
   - Sistem, bir hastadan geliyormuş gibi EKG sinyallerini alır.
   - Sinyaller belirli "pencere" boyutlarına (örneğin 750 bazlık örnekler) bölünerek kayan pencere (sliding window) yöntemiyle `TFLite` modelinden geçirilir.
   - Model her bir bölüm için 0 ile 1 arasında bir "Anomali / Risk Skoru" üretir.

3. **Risk Değerlendirmesi ve Görselleştirme:**
   - Eğer anomali skoru kritik eşiği (örn: 0.6) aşarsa, sistem bunu arayüz üzerinde Matplotlib kullanarak EKG grafiği üzerinde **oklar ve skorlarla** nokta atışı işaretler.
   - Hastanın vital bulgularıyla birlikte tüm EKG analiz sonuçları birleştirilir ve **"Genel Kalp Krizi Riski"** (Düşük Risk, Orta Risk, Kritik Tehlike vb.) olarak ekrana yansıtılır.
   - Tüm analiz aşamaları arayüzdeki "Analiz Günlüğü" sekmesine aktarılır.

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8+
- Gerekli kütüphaneleri yüklemek için:
  ```bash
  pip install -r requirements.txt
  ```
  > `tensorflow` yalnızca modeli sıfırdan eğitmek için gereklidir.
  > Dashboard (`main.py`) TensorFlow bağımlılığı olmadan çalışır.

### 📁 Büyük Dosyalar Hakkında (Önemli)

**`ptbxl_database.csv`** bu repoda bulunmaz (lisans ve boyut nedeniyle `.gitignore`'da).
Eğitim için PhysioNet'ten indirin:
```
https://physionet.org/content/ptb-xl/1.0.3/
```

**Model dosyaları** (`models/*.h5`, `*.tflite`, `*.pkl`) Git LFS ile yönetilmelidir.
İlk kez:
```bash
git lfs install
git lfs migrate import --include="*.h5,*.tflite,*.pkl" --everything
```

### Kullanım

TFLite yapay zeka modelini baştan eğitmek ve kaydetmek için (Opsiyonel):
```bash
python train_ptbxl_model.py
```

RF (Vital Bulgular) modelini eğitmek için:
```bash
python train_rf_model.py
```

EKG analiz ve klinik arayüzü (Dashboard) çalıştırmak için:
```bash
python main.py
```
