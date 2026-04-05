"""
HeartGuard - Birlesik EKG Analiz Dashboard'u
"""

import threading
import random
import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.ticker as ticker

import tensorflow as tf
import wfdb
import joblib
import os

# ─── Vital Bulgular Modeli ─────────────────────────────────────────────────────
RF_MODEL_PATH = os.path.join("models", "vitals_rf_model.pkl")

def load_rf_model():
    return joblib.load(RF_MODEL_PATH)

def generate_vitals(critical=None):
    """Rastgele nabiz + tansiyon verisi uretir."""
    if critical is None:
        critical = random.random() < 0.3   # %30 ihtimalle kritik
    if critical:
        hr  = random.randint(100, 160)     # Tasikardi
        sys_bp = random.randint(150, 200)  # Yuksek TA
        dia_bp = random.randint(95, 130)
    else:
        hr  = random.randint(60, 90)       # Normal nabiz
        sys_bp = random.randint(110, 135)  # Normal TA
        dia_bp = random.randint(70, 85)
    return hr, sys_bp, dia_bp

# ─── Sabitler ─────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "ecg_model.tflite")
WINDOW      = 750
STEP        = 250
THRESHOLD   = 0.6

# Refresh'te rastgele sec: (kayit_id, sampto, baslik)
MITBIH_POOL = [
    ("100", 3600, "MIT-BIH #100 – Normal Sinus Ritmi"),
    ("101", 3600, "MIT-BIH #101 – Normal Sinus Ritmi"),
    ("103", 3600, "MIT-BIH #103 – Normal Sinus Ritmi"),
    ("200", 3600, "MIT-BIH #200 – Ventrikuler Aritmi"),
    ("201", 3600, "MIT-BIH #201 – Supraventrikuler Aritmi"),
    ("202", 3600, "MIT-BIH #202 – Kompleks Aritmi"),
    ("207", 3600, "MIT-BIH #207 – Ventrikuler Fibrilasyon"),
    ("208", 3600, "MIT-BIH #208 – PVC + Normal"),
]

BG       = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_BG  = "#0f3460"
ACCENT   = "#e94560"
GREEN    = "#00b894"
YELLOW   = "#fdcb6e"
FG       = "#e0e0e0"
FONT_MONO = ("Consolas", 10)
FONT_HEAD = ("Segoe UI", 13, "bold")
FONT_SUB  = ("Segoe UI", 10)

# ─── Risk Kategorisi ──────────────────────────────────────────────────────────
def risk_label(score):
    pct = score * 100
    if pct < 10:   return pct, "COK DUSUK RISK",   "#00b894"
    elif pct < 35: return pct, "DUSUK RISK",        "#55efc4"
    elif pct < 60: return pct, "ORTA RISK",         "#fdcb6e"
    elif pct < 80: return pct, "YUKSEK RISK",       "#e17055"
    else:          return pct, "KRITIK TEHLIKE!",    "#e94560"

# ─── Model ────────────────────────────────────────────────────────────────────
def load_interpreter():
    i = tf.lite.Interpreter(model_path=MODEL_PATH)
    i.allocate_tensors()
    return i

def detect_anomalies(signal, interp, fs):
    inp  = interp.get_input_details()
    outp = interp.get_output_details()
    results = []
    for start in range(0, len(signal) - WINDOW, STEP):
        chunk = signal[start:start + WINDOW].reshape(1, WINDOW, 1).astype(np.float32)
        interp.set_tensor(inp[0]['index'], chunk)
        interp.invoke()
        score = float(interp.get_tensor(outp[0]['index'])[0][0])
        results.append(((start + WINDOW // 2) / fs, score))
    return results

# ─── Veri Kaynakları ──────────────────────────────────────────────────────────
def fetch_mitbih(record_id, sampto=3600):
    rec = wfdb.rdrecord(str(record_id), sampto=sampto, pn_dir="mitdb")
    sig = rec.p_signal[:, 0]
    fs  = rec.fs
    return sig, np.arange(len(sig)) / fs, fs

def make_stemi(duration=4, fs=250):
    n   = fs * duration
    t   = np.linspace(0, duration, n)
    sig = np.random.randn(n) * 0.035
    for beat in np.arange(0.25, duration, 0.82):
        i = int(beat * fs)
        if i + 110 > n: break
        sig[i:i+15]    += np.linspace(0, 0.14, 15)
        sig[i+15:i+30] += np.linspace(0.14, 0, 15)
        sig[i+30:i+35] += np.linspace(0, -0.10, 5)
        sig[i+35:i+40] += np.linspace(-0.10, 1.45, 5)
        sig[i+40:i+45] += np.linspace(1.45, 0.0, 5)
        sig[i+45:i+65] += 0.38
        if i+90  < n: sig[i+65:i+90]  += np.linspace(0.38, 0.55, 25)
        if i+110 < n: sig[i+90:i+110] += np.linspace(0.55, 0, 20)
    return sig, t, fs

# ─── Grafik ───────────────────────────────────────────────────────────────────
def draw_ecg(ax, sig, t, flags, title):
    ax.set_facecolor("#ffe8e8")
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.grid(which="minor", color="#f4a0a0", linewidth=0.2, alpha=0.4)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(which="major", color="#cc3333", linewidth=0.6, alpha=0.3)
    ax.plot(t, sig, color="#111111", linewidth=0.8, zorder=3)
    ax.axhline(0, color="#880000", linewidth=0.8, linestyle="--", alpha=0.5)

    sig_max = float(np.max(np.abs(sig)))
    arrow_y = sig_max + 0.3
    annotated = set()
    for (tc, score) in flags:
        if score < THRESHOLD: continue
        rk = round(tc, 1)
        if rk in annotated: continue
        annotated.add(rk)
        idx = min(int(tc * len(sig) / t[-1]), len(sig) - 1)
        ax.annotate(f"Anomali {score:.2f}",
                    xy=(tc, sig[idx]), xytext=(tc, arrow_y),
                    fontsize=7.5, fontweight="bold", color="#cc0000", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.6),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff0f0", ec="#cc0000", alpha=0.9),
                    zorder=5)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(float(np.min(sig)) - 0.2, arrow_y + 0.35)
    ax.set_title(title, fontsize=9, fontweight="bold", color="#111111", pad=4)
    ax.set_xlabel("Sure (sn)", fontsize=8, color="#333")
    ax.set_ylabel("mV", fontsize=8, color="#333")
    ax.tick_params(labelsize=7, colors="#444")
    for sp in ax.spines.values():
        sp.set_edgecolor("#990000"); sp.set_linewidth(0.8)

# ─── Dashboard ────────────────────────────────────────────────────────────────
class HeartGuardDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HeartGuard - EKG Analiz Dashboard")
        self.configure(bg=BG)
        try:
            self.state("zoomed")
        except Exception:
            self.attributes("-zoomed", True)
        
        # Performance: Load models once at startup
        self.status_var = tk.StringVar(value="Modeller yukleniyor...")
        self.interp = load_interpreter()
        self.rf_model = load_rf_model()
        
        self._build_ui()
        self.after(200, self._start_analysis)

    def _build_ui(self):
        # ── Baslik ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=ACCENT, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="HeartGuard  -  Gercek Zamanli EKG Analiz Sistemi",
                 font=("Segoe UI", 15, "bold"), bg=ACCENT, fg="white").pack(side="left", padx=16)
        self.time_lbl = tk.Label(hdr,
                 text=f"Analiz Zamani: {datetime.now().strftime('%d.%m.%Y  %H:%M:%S')}",
                 font=FONT_SUB, bg=ACCENT, fg="#fff0f0")
        self.time_lbl.pack(side="right", padx=16)
        self.btn_refresh = tk.Button(
            hdr, text="  Yeni Veri Olustur",
            font=("Segoe UI", 10, "bold"), bg="white", fg=ACCENT,
            activebackground="#ffcdd2", activeforeground="#b71c1c",
            relief="flat", padx=14, pady=5, cursor="hand2",
            command=self._refresh_all)
        self.btn_refresh.pack(side="right", padx=8)

        # ── Ana Govde ─────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Sol: Grafikler
        left = tk.Frame(body, bg=PANEL_BG, bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        tk.Label(left, text="EKG Grafikleri", font=FONT_HEAD, bg=PANEL_BG, fg=ACCENT).pack(pady=(8, 2))

        self.fig = Figure(figsize=(10, 7), facecolor="#f8f8f8")
        self.fig.subplots_adjust(hspace=0.45)
        self.ax1, self.ax2, self.ax3 = self.fig.subplots(3, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        # Buyut butonlari
        self.chart_data = [None, None, None]   # (sig, t, flags, title) her slot icin
        btn_bar = tk.Frame(left, bg=PANEL_BG)
        btn_bar.pack(fill="x", padx=6, pady=(0, 6))
        for i, lbl in enumerate(["Grafik 1 - Buyut", "Grafik 2 - Buyut", "STEMI - Buyut"]):
            slot = i
            tk.Button(btn_bar, text=f"  {lbl}",
                      font=("Segoe UI", 8, "bold"),
                      bg=TEXT_BG, fg=FG,
                      activebackground=ACCENT, activeforeground="white",
                      relief="flat", padx=8, pady=3, cursor="hand2",
                      command=lambda s=slot: self._open_zoom(s)
                      ).pack(side="left", padx=4)

        # Sag: Risk Analizi
        right = tk.Frame(body, bg=PANEL_BG, bd=1, relief="solid", width=360)
        right.pack(side="right", fill="y", padx=(6, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Risk Analizi", font=FONT_HEAD, bg=PANEL_BG, fg=ACCENT).pack(pady=(10, 6))

        # TEK Genel Risk Kutusu
        box = tk.Frame(right, bg=TEXT_BG, pady=16, padx=16)
        box.pack(fill="x", padx=10, pady=4)

        tk.Label(box, text="GENEL KALP KRIZI RISKI",
                 font=("Segoe UI", 10, "bold"), bg=TEXT_BG, fg="#aaaaaa").pack(anchor="w")

        self.lbl_score = tk.Label(box, text="Hesaplaniyor...",
                                  font=("Consolas", 22, "bold"), bg=TEXT_BG, fg=YELLOW)
        self.lbl_score.pack(anchor="w", pady=(6, 2))

        self.lbl_pct = tk.Label(box, text="",
                                font=("Segoe UI", 14, "bold"), bg=TEXT_BG, fg=YELLOW)
        self.lbl_pct.pack(anchor="w", pady=(0, 4))

        self.lbl_category = tk.Label(box, text="",
                                     font=("Segoe UI", 12, "bold"), bg=TEXT_BG, fg=YELLOW)
        self.lbl_category.pack(anchor="w")

        # Separator
        tk.Frame(right, bg=ACCENT, height=1).pack(fill="x", padx=10, pady=8)

        # Detay etiketi
        self.lbl_detail = tk.Label(right, text="",
                                   font=FONT_SUB, bg=PANEL_BG, fg=FG,
                                   justify="left", anchor="w", wraplength=320)
        self.lbl_detail.pack(fill="x", padx=12, pady=(0, 4))

        # ── Vital Bulgular Kutusu ─────────────────────────────────────────────
        vbox = tk.Frame(right, bg=TEXT_BG, pady=10, padx=14)
        vbox.pack(fill="x", padx=10, pady=4)

        tk.Label(vbox, text="VITAL BULGULAR",
                 font=("Segoe UI", 9, "bold"), bg=TEXT_BG, fg="#aaaaaa").pack(anchor="w")

        row1 = tk.Frame(vbox, bg=TEXT_BG)
        row1.pack(fill="x", pady=(6, 0))
        tk.Label(row1, text="Nabiz:", font=("Segoe UI", 10), bg=TEXT_BG, fg=FG, width=12, anchor="w").pack(side="left")
        self.lbl_hr = tk.Label(row1, text="-- bpm", font=("Consolas", 11, "bold"), bg=TEXT_BG, fg=YELLOW)
        self.lbl_hr.pack(side="left")

        row2 = tk.Frame(vbox, bg=TEXT_BG)
        row2.pack(fill="x", pady=2)
        tk.Label(row2, text="Sistolik TA:", font=("Segoe UI", 10), bg=TEXT_BG, fg=FG, width=12, anchor="w").pack(side="left")
        self.lbl_sys = tk.Label(row2, text="-- mmHg", font=("Consolas", 11, "bold"), bg=TEXT_BG, fg=YELLOW)
        self.lbl_sys.pack(side="left")

        row3 = tk.Frame(vbox, bg=TEXT_BG)
        row3.pack(fill="x", pady=2)
        tk.Label(row3, text="Diastolik TA:", font=("Segoe UI", 10), bg=TEXT_BG, fg=FG, width=12, anchor="w").pack(side="left")
        self.lbl_dia = tk.Label(row3, text="-- mmHg", font=("Consolas", 11, "bold"), bg=TEXT_BG, fg=YELLOW)
        self.lbl_dia.pack(side="left")

        row4 = tk.Frame(vbox, bg=TEXT_BG)
        row4.pack(fill="x", pady=(6, 0))
        tk.Label(row4, text="RF Tahmini:", font=("Segoe UI", 10), bg=TEXT_BG, fg=FG, width=12, anchor="w").pack(side="left")
        self.lbl_rf = tk.Label(row4, text="--", font=("Segoe UI", 11, "bold"), bg=TEXT_BG, fg=YELLOW)
        self.lbl_rf.pack(side="left")

        # Log
        tk.Label(right, text="Analiz Gunlugu", font=FONT_HEAD, bg=PANEL_BG, fg=ACCENT).pack(pady=(8, 4))
        self.log = scrolledtext.ScrolledText(right, font=FONT_MONO, bg="#0a1628",
                                             fg=FG, insertbackground=FG,
                                             state="disabled", wrap="word", bd=0)
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log.tag_config("ok",   foreground=GREEN)
        self.log.tag_config("warn", foreground=ACCENT)
        self.log.tag_config("info", foreground=YELLOW)
        self.log.tag_config("head", foreground="#64b5f6", font=("Consolas", 10, "bold"))

        # Alt Durum
        self.status_var = tk.StringVar(value="Baslatiliyor...")
        tk.Label(self, textvariable=self.status_var,
                 font=FONT_SUB, bg="#0d0d1a", fg=YELLOW, anchor="w", padx=10).pack(fill="x", side="bottom")

    def _open_zoom(self, slot):
        """Secilen EKG grafinini ayri, buyuk bir pencerede acar (zoom/pan destekli)."""
        data = self.chart_data[slot]
        if data is None:
            return
        sig, t, flags, title = data
        win = tk.Toplevel(self)
        win.title(f"EKG Detay  —  {title}")
        win.configure(bg=BG)
        win.geometry("1200x620")
        fig2 = Figure(figsize=(15, 5), facecolor="#f8f8f8")
        ax2  = fig2.add_subplot(1, 1, 1)
        draw_ecg(ax2, sig, t, flags, title)
        fig2.tight_layout(pad=1.5)
        canv = FigureCanvasTkAgg(fig2, master=win)
        toolbar = NavigationToolbar2Tk(canv, win)
        toolbar.update()
        canv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
        canv.draw()

    def _log(self, msg, tag="info"):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def _update_risk(self, max_score, total_anomalies, sources):
        pct, category, color = risk_label(max_score)
        self.lbl_score.config(text=f"Risk: {max_score:.4f}", fg=color)
        self.lbl_pct.config(text=f"Kalp Krizi Olasılığı:  %{pct:.1f}", fg=color)
        self.lbl_category.config(text=category, fg=color)
        detail = (
            f"Toplam Anomali: {total_anomalies} bolge tespit edildi\n"
            f"Analiz edilen: {', '.join(sources)}"
        )
        self.lbl_detail.config(text=detail)

    def _refresh_all(self):
        self.btn_refresh.config(state="disabled", text="  Analiz Ediliyor...")
        self.time_lbl.config(text=f"Analiz Zamani: {datetime.now().strftime('%d.%m.%Y  %H:%M:%S')}")
        self.lbl_score.config(text="Hesaplaniyor...", fg=YELLOW)
        self.lbl_pct.config(text="")
        self.lbl_category.config(text="")
        self.lbl_detail.config(text="")
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.cla()
        self.canvas.draw()
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _start_analysis(self):
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        self._log("=" * 42, "head")
        self._log(" HeartGuard EKG Analiz Basladi", "head")
        self._log("=" * 42 + "\n", "head")
        try:
            self.status_var.set("Analiz baslatildi...")
            # Models are already cached in self.interp and self.rf_model
            interp = self.interp
            rf = self.rf_model

            # ── Vital Bulgular (RF Modeli) ─────────────────────────────────────
            self._log("\n--- Vital Bulgular Analizi ---", "head")
            rf = load_rf_model()
            hr, sys_bp, dia_bp = generate_vitals()
            import pandas as pd
            X_vitals = pd.DataFrame([[hr, sys_bp, dia_bp]],
                                    columns=["HeartRate", "SystolicBP", "DiastolicBP"])
            rf_pred  = rf.predict(X_vitals)[0]          # 0=Normal 1=Kritik
            rf_prob  = rf.predict_proba(X_vitals)[0][1] # Kritik olasılığı
            self._log(f"Nabiz   : {hr} bpm", "info")
            self._log(f"Sistolik: {sys_bp} mmHg", "info")
            self._log(f"Diastolik: {dia_bp} mmHg", "info")
            rf_label = "KRITIK" if rf_pred == 1 else "Normal"
            rf_color = ACCENT if rf_pred == 1 else GREEN
            self._log(f"RF Tahmini: {rf_label} (olasilik: {rf_prob:.2f})\n",
                      "warn" if rf_pred == 1 else "ok")

            def _upd_vitals(h=hr, s=sys_bp, d=dia_bp, lbl=rf_label, col=rf_color):
                hc = ACCENT if h > 100 or h < 50 else GREEN
                sc = ACCENT if s > 140 else GREEN
                dc = ACCENT if d > 90  else GREEN
                self.lbl_hr.config(text=f"{h} bpm", fg=hc)
                self.lbl_sys.config(text=f"{s} mmHg", fg=sc)
                self.lbl_dia.config(text=f"{d} mmHg", fg=dc)
                self.lbl_rf.config(text=lbl, fg=col)
            self.after(0, _upd_vitals)

            all_max   = 0.0
            total_anom = 0
            sources   = []

            # ── Rastgele 2 MIT-BIH kaydi sec ─────────────────────────────────
            chosen = random.sample(MITBIH_POOL, 2)
            ax_list = [self.ax1, self.ax2]

            for (rec_id, sampto, rec_title), ax in zip(chosen, ax_list):
                self.status_var.set(f"MIT-BIH Kayit {rec_id} indiriliyor...")
                self._log(f"--- {rec_title} ---", "head")
                sig, t, fs = fetch_mitbih(rec_id, sampto)
                self._log(f"Uzunluk: {t[-1]:.1f} sn | {len(sig)} ornek | {fs} Hz", "info")
                flags = detect_anomalies(sig, interp, fs)
                crit  = [s for _, s in flags if s > THRESHOLD]
                mx    = max((s for _, s in flags), default=0)
                self._log(f"Anomali: {len(crit)} | Maks skor: {mx:.4f}\n",
                           "warn" if crit else "ok")
                all_max    = max(all_max, mx)
                total_anom += len(crit)
                sources.append(f"#{rec_id}")
                slot_idx = list(chosen).index((rec_id, sampto, rec_title))
                self.chart_data[slot_idx] = (sig, t, flags, rec_title)
                _ax = ax
                _s, _t, _f, _ti = sig, t, flags, rec_title
                self.after(0, lambda a=_ax, s=_s, ti=_t, fl=_f, tt=_ti:
                           (draw_ecg(a, s, ti, fl, tt), self.canvas.draw()))

            # ── Sentetik STEMI ────────────────────────────────────────────────
            self.status_var.set("Sentetik STEMI olusturuluyor...")
            self._log("--- Sentetik EKG - ST Yukselmesi (STEMI) ---", "head")
            sig3, t3, fs3 = make_stemi()
            self._log(f"Uzunluk: {t3[-1]:.1f} sn | {len(sig3)} ornek | {fs3} Hz", "info")
            flags3 = detect_anomalies(sig3, interp, fs3)
            crit3  = [s for _, s in flags3 if s > THRESHOLD]
            mx3    = max((s for _, s in flags3), default=0)
            self._log(f"Anomali: {len(crit3)} | Maks skor: {mx3:.4f}\n",
                      "warn" if crit3 else "ok")
            all_max    = max(all_max, mx3)
            total_anom += len(crit3)
            sources.append("STEMI-sim")
            stemi_title = "Sentetik EKG - ST Yukselmesi / STEMI"
            self.chart_data[2] = (sig3, t3, flags3, stemi_title)
            self.after(0, lambda s=sig3, t=t3, f=flags3, tt=stemi_title:
                       (draw_ecg(self.ax3, s, t, f, tt), self.canvas.draw()))

            # ── Ozet ──────────────────────────────────────────────────────────
            self._log("=" * 42, "head")
            self._log(" GENEL DEGERLENDIRME", "head")
            self._log("=" * 42, "head")
            pct, cat, _ = risk_label(all_max)
            self._log(f"Maks. Risk Skoru : {all_max:.4f}", "info")
            self._log(f"Toplam Anomali   : {total_anom}", "warn" if total_anom else "ok")
            self._log(f"Kategori         : {cat}", "warn" if all_max >= THRESHOLD else "ok")
            self._log(f"\nAnaliz tamamlandi: {datetime.now().strftime('%H:%M:%S')}", "info")

            self.after(0, lambda: self._update_risk(all_max, total_anom, sources))
            self.after(100, self.canvas.draw) # Final composite draw
            self.status_var.set("Analiz tamamlandi")
            self.after(0, lambda: self.btn_refresh.config(
                state="normal", text="  Yeni Veri Olustur"))

        except Exception as e:
            self._log(f"\nHATA: {e}", "warn")
            self.status_var.set(f"Hata: {e}")
            self.after(0, lambda: self.btn_refresh.config(
                state="normal", text="  Yeni Veri Olustur"))


if __name__ == "__main__":
    app = HeartGuardDashboard()
    app.mainloop()
