"""
OSA Detect — Backend Inference Server
รับไฟล์เสียง (.m4a/.wav) จากแอป
decode → Log-Mel Spectrogram → CNN inference → ส่งผลกลับ
"""

import os
import io
import json
import tempfile
import numpy as np
import torch
import torch.nn as nn
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ============================================================
# Config (ต้อง match กับ Colab training)
# ============================================================
SAMPLE_RATE   = 16000
CLIP_DURATION = 30.0
N_MELS        = 64
N_FFT         = 1024
HOP_LENGTH    = 512
SAMPLES       = int(SAMPLE_RATE * CLIP_DURATION)

APNEA_SILENCE_MIN = 10.0   # วินาที (AASM standard)
CLASS_NORMAL  = 0
CLASS_SNORING = 1
CLASS_APNEA   = 2

# ============================================================
# Model Architecture (ต้อง match กับ training)
# ============================================================
class DWSConv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU6(inplace=True)
        )
    def forward(self, x): return self.net(x)

class OSAModel(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            DWSConv(32, 64),
            DWSConv(64, 128, stride=2),
            DWSConv(128, 128),
            DWSConv(128, 256, stride=2),
            DWSConv(256, 256),
            DWSConv(256, 512, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

# ============================================================
# โหลด model ตอน startup
# ============================================================
model = None

def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), 'osa_best.pt')
    if not os.path.exists(model_path):
        print('⚠️ ไม่พบ osa_best.pt — inference จะไม่ทำงาน')
        return
    model = OSAModel(n_classes=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f'✅ โหลด model สำเร็จ')

# ============================================================
# Preprocessing
# ============================================================
def audio_to_logmel(y, sr=SAMPLE_RATE):
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ============================================================
# วิเคราะห์ audio file ทั้งคืน
# แบ่งเป็น clip 30 วิ แล้ว infer แต่ละ clip
# ============================================================
def analyze_audio(audio_bytes, filename='audio.m4a'):
    # บันทึกไฟล์ชั่วคราว
    suffix = os.path.splitext(filename)[-1] or '.m4a'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        # โหลด audio ทั้งไฟล์
        y_full, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        total_duration = len(y_full) / SAMPLE_RATE
        print(f'Audio duration: {total_duration:.1f} วินาที')

        # แบ่งเป็น clip 30 วิ (overlap 50% เพื่อไม่พลาด event)
        clip_hop     = int(SAMPLE_RATE * CLIP_DURATION * 0.5)
        clip_samples = SAMPLES
        clips        = []
        timestamps   = []

        offset = 0
        while offset + clip_samples <= len(y_full):
            clips.append(y_full[offset:offset + clip_samples])
            timestamps.append(offset / SAMPLE_RATE)
            offset += clip_hop

        # ถ้าไม่มี clip เพียงพอ ใช้ทั้งไฟล์
        if len(clips) == 0:
            clips.append(y_full)
            timestamps.append(0)

        print(f'วิเคราะห์ {len(clips)} clips')

        events    = []
        clip_preds = []

        for i, (clip, t_start) in enumerate(zip(clips, timestamps)):
            spec    = audio_to_logmel(clip)
            tensor  = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor).numpy()[0]

            probs     = softmax(logits)
            predicted = int(np.argmax(probs))
            clip_preds.append(predicted)

            confidence = float(probs[predicted])
            t_str = f'{int(t_start//3600):02d}:{int((t_start%3600)//60):02d}:{int(t_start%60):02d}'

            if predicted == CLASS_APNEA and confidence >= 0.6:
                events.append({
                    'type':       'apnea',
                    'time':       t_str,
                    'timestamp':  float(t_start),
                    'confidence': round(confidence * 100, 1),
                    'msg':        f'⚠️ หยุดหายใจ ({confidence*100:.0f}%)',
                })
            elif predicted == CLASS_SNORING and confidence >= 0.6:
                events.append({
                    'type':       'snore',
                    'time':       t_str,
                    'timestamp':  float(t_start),
                    'confidence': round(confidence * 100, 1),
                    'msg':        f'🔊 เสียงกรน ({confidence*100:.0f}%)',
                })

        # คำนวณ AHI
        apnea_count  = sum(1 for e in events if e['type'] == 'apnea')
        snore_count  = sum(1 for e in events if e['type'] == 'snore')
        sleep_hours  = total_duration / 3600
        ahi          = round(apnea_count / max(sleep_hours, 1/60), 1)

        # จำแนกระดับความเสี่ยง
        if ahi < 5:
            risk = 'ปกติ'
        elif ahi < 15:
            risk = 'เล็กน้อย'
        elif ahi < 30:
            risk = 'ปานกลาง'
        else:
            risk = 'รุนแรง'

        return {
            'success':      True,
            'duration':     round(total_duration),
            'ahi':          ahi,
            'riskLabel':    risk,
            'apneaCount':   apnea_count,
            'snoreCount':   snore_count,
            'events':       events,
            'totalClips':   len(clips),
            'engine':       'ai-server',
        }

    finally:
        os.unlink(tmp_path)

# ============================================================
# API Routes
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':      'ok',
        'modelLoaded': model is not None,
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return jsonify({'success': False, 'error': 'Model ยังไม่ถูกโหลด'}), 503

    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'ไม่พบไฟล์เสียง'}), 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    if len(audio_bytes) == 0:
        return jsonify({'success': False, 'error': 'ไฟล์เสียงว่าง'}), 400

    print(f'รับไฟล์: {audio_file.filename} ({len(audio_bytes)/1024:.0f} KB)')

    try:
        result = analyze_audio(audio_bytes, audio_file.filename)
        return jsonify(result)
    except Exception as e:
        print(f'Error: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
