"""
OSA Detect — Backend API Server
Render.com + PostgreSQL
"""

import os, json, tempfile, hashlib
import numpy as np
import torch
import torch.nn as nn
import librosa
import bcrypt
import jwt
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from functools import wraps

app = Flask(__name__)
CORS(app)

# ============================================================
# Config
# ============================================================
DATABASE_URL    = os.environ.get('DATABASE_URL', '')
JWT_SECRET      = os.environ.get('JWT_SECRET', 'osa-detect-secret-2024')
JWT_EXPIRE_DAYS = 30

SAMPLE_RATE   = 16000
CLIP_DURATION = 10.0
N_MELS        = 64
N_FFT         = 1024
HOP_LENGTH    = 512
SAMPLES       = int(SAMPLE_RATE * CLIP_DURATION)

# ============================================================
# Database
# ============================================================
def get_db():
    if 'db' not in g:
        url = DATABASE_URL
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
        g.db = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def query(sql, params=None, fetch='all'):
    db  = get_db()
    cur = db.cursor()
    cur.execute(sql, params or ())
    db.commit()
    if fetch == 'one':  return cur.fetchone()
    if fetch == 'none': return cur.rowcount
    return cur.fetchall()

def init_db():
    try:
        url = DATABASE_URL
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
        conn = psycopg2.connect(url)
        cur  = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                email         VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                first_name    VARCHAR(100),
                last_name     VARCHAR(100),
                gender        VARCHAR(20),
                age           INTEGER,
                conditions    TEXT,
                created_at    TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS sleep_sessions (
                id            SERIAL PRIMARY KEY,
                user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
                date          DATE NOT NULL,
                duration_sec  INTEGER DEFAULT 0,
                ahi           NUMERIC(5,1) DEFAULT 0,
                risk_label    VARCHAR(20) DEFAULT 'ปกติ',
                apnea_count   INTEGER DEFAULT 0,
                snore_count   INTEGER DEFAULT 0,
                move_count    INTEGER DEFAULT 0,
                engine        VARCHAR(20) DEFAULT 'dsp',
                wellness_pct  INTEGER DEFAULT 0,
                created_at    TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS sleep_events (
                id         SERIAL PRIMARY KEY,
                session_id INTEGER REFERENCES sleep_sessions(id) ON DELETE CASCADE,
                type       VARCHAR(20),
                time_str   VARCHAR(10),
                timestamp  NUMERIC(10,1),
                msg        TEXT,
                confidence NUMERIC(5,1)
            );
            CREATE TABLE IF NOT EXISTS surveys (
                id           SERIAL PRIMARY KEY,
                session_id   INTEGER REFERENCES sleep_sessions(id) ON DELETE CASCADE,
                wellness_pct INTEGER DEFAULT 0,
                answers      TEXT,
                created_at   TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS preferences (
                id                  SERIAL PRIMARY KEY,
                user_id             INTEGER REFERENCES users(id) ON DELETE CASCADE UNIQUE,
                smart_alarm_enabled BOOLEAN DEFAULT TRUE,
                smart_alarm_time    VARCHAR(5) DEFAULT '06:30',
                sleep_goal_hours    NUMERIC(3,1) DEFAULT 8.0,
                updated_at          TIMESTAMP DEFAULT NOW()
            );
        ''')
        conn.commit()
        conn.close()
        print('✅ Database tables ready')
    except Exception as e:
        print(f'❌ Database init error: {e}')

# ============================================================
# AI Model
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
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            DWSConv(32, 64), DWSConv(64, 128, stride=2),
            DWSConv(128, 128), DWSConv(128, 256, stride=2),
            DWSConv(256, 256), DWSConv(256, 512, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU6(),
            nn.Dropout(0.2), nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

model = None

def load_model():
    global model
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'osa_best.pt')
    print(f'[MODEL] path={path} exists={os.path.exists(path)}')
    if not os.path.exists(path):
        return
    try:
        # binary model: class 0 = Normal, class 1 = Apnea
        model = OSAModel(n_classes=2)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        print('✅ Model loaded')
    except Exception as e:
        print(f'❌ Model load error: {e}')
        model = None

# ============================================================
# Auth helpers
# ============================================================
def make_token(user_id):
    return jwt.encode(
        {'user_id': user_id, 'exp': datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)},
        JWT_SECRET, algorithm='HS256'
    )

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'ต้องการ token'}), 401
        try:
            payload   = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            g.user_id = payload['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token หมดอายุ'}), 401
        except Exception:
            return jsonify({'error': 'Token ไม่ถูกต้อง'}), 401
        return f(*args, **kwargs)
    return decorated

# ============================================================
# Audio analysis
# ============================================================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def audio_to_logmel(y):
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]
    mel     = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)

def load_audio_lowmem(path):
    """โหลด audio แบบประหยัด RAM — แปลง m4a เป็น wav ก่อนด้วย ffmpeg ถ้ามี"""
    import subprocess, shutil
    wav_path = path + '.wav'
    try:
        # ลอง ffmpeg ก่อน (ถ้ามี) — ใช้ RAM น้อยกว่า audioread
        if shutil.which('ffmpeg'):
            subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-ar', str(SAMPLE_RATE), '-ac', '1', wav_path],
                capture_output=True, timeout=30
            )
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            return y, sr
    except Exception:
        pass
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
    # fallback: librosa โดยตรง
    return librosa.load(path, sr=SAMPLE_RATE, mono=True)

def analyze_audio(audio_bytes, filename='audio.m4a'):
    suffix = os.path.splitext(filename)[-1] or '.m4a'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        y_full, _      = load_audio_lowmem(tmp_path)
        total_duration = len(y_full) / SAMPLE_RATE

        # chunk นี้ควรสั้น (30 วิ) แต่ป้องกัน OOM ถ้าส่งมายาวเกิน
        MAX_DURATION = 60.0
        if total_duration > MAX_DURATION:
            print(f'[WARN] chunk too long ({total_duration:.1f}s), trimming to {MAX_DURATION}s')
            y_full = y_full[:int(MAX_DURATION * SAMPLE_RATE)]
            total_duration = MAX_DURATION

        clip_hop       = int(SAMPLE_RATE * CLIP_DURATION * 0.5)
        clips, timestamps = [], []
        offset = 0
        while offset + SAMPLES <= len(y_full):
            clips.append(y_full[offset:offset+SAMPLES])
            timestamps.append(offset / SAMPLE_RATE)
            offset += clip_hop
        if not clips:
            clips.append(y_full)
            timestamps.append(0)

        # ล้าง y_full ออกจาก RAM ก่อน inference
        del y_full

        print(f'[ANALYZE] duration={total_duration:.1f}s clips={len(clips)} CLIP_DURATION={CLIP_DURATION}')
        events = []
        for clip, t_start in zip(clips, timestamps):
            spec   = audio_to_logmel(clip)
            tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor).numpy()[0]
            probs     = softmax(logits)
            predicted = int(np.argmax(probs))
            conf      = float(probs[predicted])
            t_str     = f'{int(t_start//3600):02d}:{int((t_start%3600)//60):02d}:{int(t_start%60):02d}'

            print(f'[CLIP] t={t_start:.0f}s logits={[round(float(l),3) for l in logits.tolist()]} probs={[round(float(p),3) for p in probs.tolist()]} predicted={predicted} conf={conf:.3f}')

            # model เป็น binary: CLASS_NORMAL=0, CLASS_APNEA=1
            conf_threshold = 0.40
            if predicted == 1 and conf >= conf_threshold:
                events.append({'type': 'apnea', 'time': t_str, 'timestamp': float(t_start),
                               'confidence': round(conf*100,1), 'msg': f'หยุดหายใจ ({conf*100:.0f}%)'})

        apnea_count = sum(1 for e in events if e['type'] == 'apnea')
        snore_count = 0
        ahi         = round(apnea_count / max(total_duration/3600, 1/60), 1)
        risk        = 'ปกติ' if ahi < 5 else 'เล็กน้อย' if ahi < 15 else 'ปานกลาง' if ahi < 30 else 'รุนแรง'

        return {'success': True, 'duration': round(total_duration), 'ahi': ahi,
                'riskLabel': risk, 'apneaCount': apnea_count, 'snoreCount': snore_count,
                'events': events, 'engine': 'ai-server'}
    finally:
        os.unlink(tmp_path)

# ============================================================
# Routes
# ============================================================
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'modelLoaded': model is not None})

# --- Auth ---
@app.route('/auth/signup', methods=['POST'])
def signup():
    data     = request.json or {}
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({'error': 'กรุณากรอกอีเมลและรหัสผ่าน'}), 400
    if query('SELECT id FROM users WHERE email=%s', (email,), fetch='one'):
        return jsonify({'error': 'อีเมลนี้ถูกใช้แล้ว'}), 409
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user    = query(
        'INSERT INTO users (email,password_hash,first_name,last_name,gender,age,conditions) '
        'VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id,email,first_name,last_name,gender,age,conditions',
        (email, pw_hash, data.get('firstName'), data.get('lastName'),
         data.get('gender'), data.get('age'), data.get('conditions')), fetch='one')
    return jsonify({'success': True, 'token': make_token(user['id']), 'user': dict(user)})

@app.route('/auth/login', methods=['POST'])
def login():
    data     = request.json or {}
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    user     = query('SELECT * FROM users WHERE email=%s', (email,), fetch='one')
    if not user:
        return jsonify({'error': 'ไม่พบบัญชีนี้ในระบบ'}), 404
    if not bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
        return jsonify({'error': 'รหัสผ่านไม่ถูกต้อง'}), 401
    public = {k: v for k, v in user.items() if k != 'password_hash'}
    return jsonify({'success': True, 'token': make_token(user['id']), 'user': public})

@app.route('/auth/me')
@require_auth
def me():
    user = query('SELECT id,email,first_name,last_name,gender,age,conditions FROM users WHERE id=%s',
                 (g.user_id,), fetch='one')
    return jsonify({'success': True, 'user': dict(user)})

@app.route('/auth/profile', methods=['PUT'])
@require_auth
def update_profile():
    data = request.json or {}
    # ✅ เพิ่ม gender ที่หายไป
    query(
        'UPDATE users SET first_name=%s, last_name=%s, gender=%s, age=%s, conditions=%s WHERE id=%s',
        (data.get('firstName'), data.get('lastName'), data.get('gender'),
         data.get('age'), data.get('conditions'), g.user_id),
        fetch='none'
    )
    user = query(
        'SELECT id,email,first_name,last_name,gender,age,conditions FROM users WHERE id=%s',
        (g.user_id,), fetch='one'
    )
    return jsonify({'success': True, 'user': dict(user)})

# --- Sessions ---
@app.route('/sessions', methods=['POST'])
@require_auth
def save_session():
    data    = request.json or {}
    session = query(
        'INSERT INTO sleep_sessions (user_id,date,duration_sec,ahi,risk_label,apnea_count,snore_count,move_count,engine,wellness_pct) '
        'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id',
        (g.user_id, data.get('date', datetime.now().strftime('%Y-%m-%d')),
         data.get('duration',0), data.get('ahi',0), data.get('riskLabel','ปกติ'),
         data.get('apneaCount',0), data.get('snoreCount',0), data.get('moveCount',0),
         data.get('engine','dsp'), data.get('wellnessPct',0)), fetch='one')
    sid = session['id']
    for ev in (data.get('events') or []):
        query('INSERT INTO sleep_events (session_id,type,time_str,timestamp,msg,confidence) VALUES (%s,%s,%s,%s,%s,%s)',
              (sid, ev.get('type'), ev.get('time'), ev.get('timestamp'), ev.get('msg'), ev.get('confidence')), fetch='none')
    survey = data.get('survey')
    if survey:
        query('INSERT INTO surveys (session_id,wellness_pct,answers) VALUES (%s,%s,%s)',
              (sid, survey.get('wellnessPercent',0), json.dumps(survey.get('answers',{}))), fetch='none')
    return jsonify({'success': True, 'sessionId': str(sid)})

@app.route('/sessions', methods=['GET'])
@require_auth
def get_sessions():
    sessions = query(
        'SELECT * FROM sleep_sessions WHERE user_id=%s ORDER BY created_at DESC LIMIT 90',
        (g.user_id,))
    result = []
    for s in sessions:
        s = dict(s)
        events = query('SELECT * FROM sleep_events WHERE session_id=%s ORDER BY timestamp', (s['id'],))
        s['events'] = [dict(e) for e in events]
        result.append(s)
    return jsonify({'success': True, 'sessions': result})

@app.route('/sessions/<int:session_id>', methods=['GET'])
@require_auth
def get_session(session_id):
    session = query('SELECT * FROM sleep_sessions WHERE id=%s AND user_id=%s', (session_id, g.user_id), fetch='one')
    if not session:
        return jsonify({'error': 'ไม่พบ session'}), 404
    s      = dict(session)
    events = query('SELECT * FROM sleep_events WHERE session_id=%s ORDER BY timestamp', (session_id,))
    s['events'] = [dict(e) for e in events]
    return jsonify({'success': True, 'session': s})

@app.route('/sessions/stats', methods=['GET'])
@require_auth
def get_stats():
    stats = query(
        'SELECT COUNT(*) as total_nights, ROUND(AVG(duration_sec/3600.0),1) as avg_sleep_hours '
        'FROM sleep_sessions WHERE user_id=%s', (g.user_id,), fetch='one')
    return jsonify({'success': True, 'stats': {'total_nights': stats['total_nights'] or 0,
                                                'avg_sleep_hours': float(stats['avg_sleep_hours'] or 0),
                                                'streak': 0}})

# --- Preferences ---
@app.route('/preferences', methods=['GET'])
@require_auth
def get_preferences():
    prefs = query('SELECT * FROM preferences WHERE user_id=%s', (g.user_id,), fetch='one')
    if not prefs:
        return jsonify({'success': True, 'preferences': {'smartAlarmEnabled': True, 'smartAlarmTime': '06:30', 'sleepGoalHours': 8.0}})
    return jsonify({'success': True, 'preferences': {
        'smartAlarmEnabled': prefs['smart_alarm_enabled'],
        'smartAlarmTime':    prefs['smart_alarm_time'],
        'sleepGoalHours':    float(prefs['sleep_goal_hours'])}})

@app.route('/preferences', methods=['PUT'])
@require_auth
def update_preferences():
    data = request.json or {}
    query('INSERT INTO preferences (user_id,smart_alarm_enabled,smart_alarm_time,sleep_goal_hours) VALUES (%s,%s,%s,%s) '
          'ON CONFLICT (user_id) DO UPDATE SET smart_alarm_enabled=%s,smart_alarm_time=%s,sleep_goal_hours=%s,updated_at=NOW()',
          (g.user_id, data.get('smartAlarmEnabled',True), data.get('smartAlarmTime','06:30'), data.get('sleepGoalHours',8.0),
           data.get('smartAlarmEnabled',True), data.get('smartAlarmTime','06:30'), data.get('sleepGoalHours',8.0)), fetch='none')
    return jsonify({'success': True})

# --- AI Inference ---
@app.route('/analyze', methods=['POST'])
@require_auth
def analyze():
    if model is None:
        return jsonify({'success': False, 'error': 'Model ยังไม่ถูกโหลด'}), 503
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'ไม่พบไฟล์เสียง'}), 400
    audio_file  = request.files['audio']
    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({'success': False, 'error': 'ไฟล์เสียงว่าง'}), 400
    try:
        return jsonify(analyze_audio(audio_bytes, audio_file.filename))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
if __name__ == '__main__':
    init_db()
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
else:
    init_db()
    load_model()
