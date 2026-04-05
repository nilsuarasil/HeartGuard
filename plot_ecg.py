import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import tensorflow as tf
import wfdb
import os

# ─── Model Yükle ──────────────────────────────────────────────────────────────
MODEL_PATH = "models/ecg_model.tflite"
WINDOW     = 750   # CNN giriş uzunluğu (örnek)
STEP       = 250   # Her kaydırma adımı (örnek)
THRESHOLD  = 0.6   # Bu değerin üzeri = kritik

def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def detect_anomalies(signal, interpreter, fs):
    """Sinyali kayan pencereyle CNN'e sokup kritik zaman noktalarını döndürür."""
    inp  = interpreter.get_input_details()
    outp = interpreter.get_output_details()
    flags = []   # (merkez_saniye, risk_skoru)

    for start in range(0, len(signal) - WINDOW, STEP):
        chunk = signal[start:start + WINDOW].reshape(1, WINDOW, 1).astype(np.float32)
        interpreter.set_tensor(inp[0]['index'], chunk)
        interpreter.invoke()
        score = float(interpreter.get_tensor(outp[0]['index'])[0][0])
        center_sec = (start + WINDOW // 2) / fs
        flags.append((center_sec, score))

    return flags

def plot_ecg_clinical(signal, time_axis, flags, title="EKG Analizi", filename="ecg_chart.png"):
    fig, ax = plt.subplots(figsize=(20, 6))
    fig.patch.set_facecolor('#ffe8e8')
    ax.set_facecolor('#ffe8e8')

    # EKG kağıdı küçük kareler
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.grid(which='minor', color='#f4a0a0', linewidth=0.5, linestyle='-', alpha=0.7)
    # EKG kağıdı büyük kareler
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(which='major', color='#cc3333', linewidth=0.9, linestyle='-', alpha=0.5)

    # Sinyal
    ax.plot(time_axis, signal, color='#111111', linewidth=1.1, label='EKG Sinyali', zorder=3)

    # İzoelektrik çizgi
    ax.axhline(0, color='#880000', linewidth=1.0, linestyle='--', alpha=0.5)

    # ─── Anomali Okları ────────────────────────────────────────────────────────
    sig_max = np.max(np.abs(signal))
    arrow_y = sig_max + 0.25            # Okların dikey konumu
    annotated = set()

    for (t_center, score) in flags:
        if score < THRESHOLD:
            continue
        # Birbirine çok yakın okları tekrar ekleme
        rounded = round(t_center, 1)
        if rounded in annotated:
            continue
        annotated.add(rounded)

        # Sinyal değeri o noktada
        idx_closest = int(t_center * len(signal) / time_axis[-1])
        idx_closest = min(idx_closest, len(signal) - 1)
        sig_y = signal[idx_closest]

        ax.annotate(
            f"⚠ ({score:.2f})",
            xy=(t_center, sig_y),
            xytext=(t_center, arrow_y),
            fontsize=8, fontweight='bold', color='#cc0000',
            ha='center', va='bottom',
            arrowprops=dict(
                arrowstyle='->', color='#cc0000',
                lw=1.8,
                connectionstyle='arc3,rad=0.0'
            ),
            bbox=dict(boxstyle='round,pad=0.2', fc='#fff0f0', ec='#cc0000', alpha=0.85),
            zorder=5
        )

    ax.set_xlabel("Süre (sn)  |  25mm/sn", fontsize=12, fontweight='bold', color='#222222')
    ax.set_ylabel("Genlik (mV)  |  10mm/mV", fontsize=12, fontweight='bold', color='#222222')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14, color='#111111')
    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_ylim(np.min(signal) - 0.3, arrow_y + 0.4)

    # Açıklama
    normal_patch  = mpatches.Patch(color='#111111', label='EKG Sinyali')
    anomaly_patch = mpatches.Patch(color='#cc0000', label=f'Anomali Tespiti (skor > {THRESHOLD})')
    ax.legend(handles=[normal_patch, anomaly_patch], loc='upper right', fontsize=9, framealpha=0.8)

    for spine in ax.spines.values():
        spine.set_edgecolor('#990000')
        spine.set_linewidth(1.2)

    ax.tick_params(colors='#333333', labelsize=9)
    fig.tight_layout(pad=1.5)

    os.makedirs("ekg_grafikleri", exist_ok=True)
    save_path = os.path.join("ekg_grafikleri", filename)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Grafik kaydedildi: {save_path}")
    return save_path


def run():
    interpreter = load_model()

    # ── 1) MIT-BIH Kayıt 100 (Normal) ────────────────────────────────────────
    print("MIT-BIH Kayıt 100 (normal) analiz ediliyor...")
    rec = wfdb.rdrecord('100', sampto=3600, pn_dir='mitdb')
    sig = rec.p_signal[:, 0]
    fs  = rec.fs
    t   = np.arange(len(sig)) / fs
    flags = detect_anomalies(sig, interpreter, fs)
    plot_ecg_clinical(sig, t, flags,
                      title="MIT-BIH Kayıt 100 – Normal Sinüs Ritmi (Anomali Tespitli)",
                      filename="mitbih_100_annotated.png")

    # ── 2) MIT-BIH Kayıt 200 (Aritmili) ─────────────────────────────────────
    print("MIT-BIH Kayıt 200 (aritmili) analiz ediliyor...")
    rec2 = wfdb.rdrecord('200', sampto=3600, pn_dir='mitdb')
    sig2 = rec2.p_signal[:, 0]
    fs2  = rec2.fs
    t2   = np.arange(len(sig2)) / fs2
    flags2 = detect_anomalies(sig2, interpreter, fs2)
    plot_ecg_clinical(sig2, t2, flags2,
                      title="MIT-BIH Kayıt 200 – Ventriküler Aritmi (Anomali Tespitli)",
                      filename="mitbih_200_annotated.png")

    # ── 3) Sentetik STEMI ─────────────────────────────────────────────────────
    print("Sentetik STEMI analiz ediliyor...")
    fs3 = 250
    n   = fs3 * 4
    t3  = np.linspace(0, 4, n)
    sig3 = np.random.randn(n) * 0.04
    for beat in np.arange(0.2, 4, 0.8):
        i = int(beat * fs3)
        if i + 100 < n:
            sig3[i:i+15]   += np.linspace(0, 0.15, 15)
            sig3[i+15:i+30] += np.linspace(0.15, 0, 15)
            sig3[i+30:i+35] += np.linspace(0, -0.1, 5)
            sig3[i+35:i+40] += np.linspace(-0.1, 1.4, 5)
            sig3[i+40:i+45] += np.linspace(1.4, 0, 5)
            sig3[i+45:i+65] += 0.35          # ST yükselmesi
            sig3[i+65:i+85] += np.linspace(0.35, 0.55, 20)
            sig3[i+85:i+105] += np.linspace(0.55, 0, 20)
    flags3 = detect_anomalies(sig3, interpreter, fs3)
    plot_ecg_clinical(sig3, t3, flags3,
                      title="Sentetik EKG – ST Yükselmesi / STEMI (Anomali Tespitli)",
                      filename="sentetik_stemi_annotated.png")

    print("\nTüm grafikler 'ekg_grafikleri' klasörüne kaydedildi.")


if __name__ == "__main__":
    run()
