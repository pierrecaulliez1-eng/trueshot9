import os
import uuid
import sqlite3
import requests
import qrcode
import numpy as np
import shutil
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from flask import (Flask, render_template, request, redirect,
                   url_for, send_file, abort, flash)
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'trueshot-dev-change-in-prod')
app.config['UPLOAD_FOLDER']    = os.path.join('static', 'uploads')
app.config['CERTIFIED_FOLDER'] = os.path.join('static', 'certified')
app.config['QR_FOLDER']        = os.path.join('static', 'qrcodes')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'mp4', 'mov', 'avi', 'mkv'}
VIDEO_EXTENSIONS   = {'mp4', 'mov', 'avi', 'mkv'}

# ─── API Sightengine (NE PAS MODIFIER) ───────────────────────────────────────
SE_USER   = os.environ.get('SIGHTENGINE_USER', '')
SE_SECRET = os.environ.get('SIGHTENGINE_SECRET', '')
print(f"[DEBUG] SE_USER loaded: {bool(SE_USER)}, SE_SECRET loaded: {bool(SE_SECRET)}")
AI_THRESHOLD = 0.60

# ─── Dossiers ─────────────────────────────────────────────────────────────────
for folder in [app.config['UPLOAD_FOLDER'],
               app.config['CERTIFIED_FOLDER'],
               app.config['QR_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# ─── Flask-Login ──────────────────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, created_at):
        self.id = str(id)
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at

@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if row:
        return User(row['id'], row['username'], row['email'],
                    row['password_hash'], row['created_at'])
    return None


# ─── Base de données ──────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect('trueshot.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    # Table utilisateurs
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE,
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TEXT NOT NULL
        )
    ''')
    # Table certifications
    conn.execute('''
        CREATE TABLE IF NOT EXISTS certifications (
            id               TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            file_type        TEXT NOT NULL,
            ai_score         REAL,
            certified        INTEGER NOT NULL DEFAULT 0,
            rejection_reason TEXT,
            certified_file   TEXT,
            qr_code_file     TEXT,
            created_at       TEXT NOT NULL,
            creator_name     TEXT,
            user_id          INTEGER
        )
    ''')
    for col in ['creator_name TEXT', 'user_id INTEGER']:
        try:
            conn.execute(f'ALTER TABLE certifications ADD COLUMN {col}')
        except Exception:
            pass
    conn.commit()
    conn.close()


# ─── Utilitaires ──────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video(filename):
    return filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

def generate_unique_id():
    conn = get_db()
    for _ in range(100):
        cert_id = str(uuid.uuid4().int)[:6]
        if not conn.execute('SELECT 1 FROM certifications WHERE id = ?', (cert_id,)).fetchone():
            conn.close()
            return cert_id
    conn.close()
    return str(uuid.uuid4().int)[:8]


# ─── Analyse Sightengine (NE PAS MODIFIER) ───────────────────────────────────
def analyze_with_sightengine(image_path):
    if not SE_SECRET:
        return None
    try:
        with open(image_path, 'rb') as f:
            resp = requests.post(
                'https://api.sightengine.com/1.0/check.json',
                files={'media': f},
                data={
                    'models': 'genai',
                    'api_user': SE_USER,
                    'api_secret': SE_SECRET,
                },
                timeout=30
            )
        data = resp.json()
        if data.get('status') == 'success':
            return float(data.get('type', {}).get('ai_generated', 0.0))
        print(f"[Sightengine] erreur API : {data}")
        return None
    except Exception as e:
        print(f"[Sightengine] exception : {e}")
        return None

def extract_video_frame(video_path):
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        t = min(clip.duration * 0.15, 5.0)
        frame = clip.get_frame(t)
        clip.close()
        img = Image.fromarray(frame.astype(np.uint8))
        tmp = f"/tmp/frame_{uuid.uuid4().hex}.jpg"
        img.save(tmp, 'JPEG')
        return tmp
    except Exception as e:
        print(f"[VideoFrame] {e}")
        return None


# ─── Badge ───────────────────────────────────────────────────────────────────
def create_badge(cert_id):
    import math
    S = 600
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = S // 2, S // 2
    R = 285
    shadow = Image.new('RGBA', (S, S), (0,0,0,0))
    ImageDraw.Draw(shadow).ellipse([cx-R+4, cy-R+4, cx+R+4, cy+R+4], fill=(0,0,0,60))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    img.paste(shadow, (0,0), shadow)
    draw.ellipse([cx-R, cy-R, cx+R, cy+R], fill=(255,255,255,255))
    draw.ellipse([cx-R, cy-R, cx+R, cy+R], outline=(15,15,15,255), width=5)
    draw.ellipse([cx-R+10, cy-R+10, cx+R-10, cy+R-10], outline=(15,15,15,80), width=1)
    try:
        font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 42)
        font_sub = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 34)
    except:
        font_title = ImageFont.load_default()
        font_sub = font_title
    def arc_text(text, font, radius, start_deg, color, spacing=1.18):
        char_widths = [(font.getbbox(c)[2] - font.getbbox(c)[0]) * spacing for c in text]
        total_angle = sum(math.degrees(w / radius) for w in char_widths)
        angle = start_deg - total_angle / 2
        for ch in text:
            cw = (font.getbbox(ch)[2] - font.getbbox(ch)[0]) * spacing
            mid_angle = angle + math.degrees(cw / radius) / 2
            a = math.radians(mid_angle)
            x = cx + radius * math.cos(a - math.pi/2)
            y = cy + radius * math.sin(a - math.pi/2)
            bb = font.getbbox(ch)
            pad = 6
            tmp = Image.new('RGBA', (bb[2]-bb[0]+pad*2, bb[3]-bb[1]+pad*2), (0,0,0,0))
            ImageDraw.Draw(tmp).text((-bb[0]+pad, -bb[1]+pad), ch, font=font, fill=color)
            tmp = tmp.rotate(-mid_angle, expand=True, resample=Image.BICUBIC)
            img.paste(tmp, (int(x - tmp.width/2), int(y - tmp.height/2)), tmp)
            angle += math.degrees(cw / radius)
    arc_text("TRUESHOT", font_title, 218, 0, (12,12,12,255))
    arc_text("HUMAN CAPTURED", font_sub, 220, 180, (12,12,12,255))
    ar = 158
    draw.arc([cx-ar, cy-ar, cx+ar, cy+ar], start=145, end=215, fill=(12,12,12,255), width=3)
    draw.arc([cx-ar, cy-ar, cx+ar, cy+ar], start=325, end=35, fill=(12,12,12,255), width=3)
    cam_w, cam_h = 170, 132
    cam_x = cx - cam_w//2
    cam_y = cy - cam_h//2 + 14
    draw.rounded_rectangle([cam_x, cam_y, cam_x+cam_w, cam_y+cam_h], radius=22, outline=(12,12,12,255), width=4)
    bw = 44
    draw.rounded_rectangle([cx-bw//2, cam_y-18, cx+bw//2, cam_y+6], radius=8, fill=(255,255,255,255), outline=(12,12,12,255), width=4)
    draw.ellipse([cx-46, cy-46+14, cx+46, cy+46+14], outline=(12,12,12,255), width=4)
    draw.ellipse([cx-36, cy-36+14, cx+36, cy+36+14], outline=(12,12,12,40), width=2)
    draw.ellipse([cx-18, cy-18+14, cx+18, cy+18+14], outline=(12,12,12,255), width=3)
    draw.ellipse([cam_x+cam_w-30, cam_y+12, cam_x+cam_w-18, cam_y+24], fill=(12,12,12,255))
    return img

def overlay_badge_on_image(src, cert_id, dst):
    img = Image.open(src).convert('RGBA')
    badge = create_badge(cert_id)
    # Badge = 12% du petit côté de la photo (net à toute taille)
    target = max(80, min(img.width, img.height) // 8)
    badge = badge.resize((target, target), Image.LANCZOS)
    x = 20
    y = img.height - badge.height - 20
    img.paste(badge, (x, y), badge)
    ext = dst.rsplit('.', 1)[1].lower()
    if ext in ('jpg', 'jpeg'):
        img.convert('RGB').save(dst, 'JPEG', quality=95)
    else:
        img.save(dst)

def overlay_badge_on_video(src, cert_id, dst):
    try:
        from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
        video = VideoFileClip(src)
        badge = create_badge(cert_id)
        target = max(80, min(video.w, video.h) // 8)
        badge = badge.resize((target, target), Image.LANCZOS)
        badge_tmp = f'/tmp/badge_{cert_id}.png'
        badge.save(badge_tmp)
        badge_clip = (ImageClip(badge_tmp)
                      .set_duration(video.duration)
                      .set_position((20, video.h - badge.height - 20)))
        final = CompositeVideoClip([video, badge_clip])
        final.write_videofile(dst, codec='libx264', audio_codec='aac',
                              verbose=False, logger=None)
        video.close()
        final.close()
        try:
            os.remove(badge_tmp)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"[VideoOverlay] {e}")
        return False

def generate_qr(cert_id, base_url):
    verify_url = f"{base_url.rstrip('/')}/verify/{cert_id}"
    qr = qrcode.QRCode(version=1,
                       error_correction=qrcode.constants.ERROR_CORRECT_H,
                       box_size=8, border=3)
    qr.add_data(verify_url)
    qr.make(fit=True)
    return qr.make_image(fill_color='#08090d', back_color='white')


# ─── Routes : comptes ─────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')

        if not username or not email or not password:
            error = 'missing_fields'
        elif password != confirm:
            error = 'password_mismatch'
        elif len(password) < 6:
            error = 'password_short'
        else:
            conn = get_db()
            if conn.execute('SELECT 1 FROM users WHERE username = ?', (username,)).fetchone():
                error = 'username_taken'
            elif conn.execute('SELECT 1 FROM users WHERE email = ?', (email,)).fetchone():
                error = 'email_taken'
            else:
                conn.execute('''
                    INSERT INTO users (username, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, generate_password_hash(password),
                      datetime.now().strftime('%d/%m/%Y à %H:%M')))
                conn.commit()
                row = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
                conn.close()
                user = User(row['id'], row['username'], row['email'],
                            row['password_hash'], row['created_at'])
                login_user(user)
                return redirect(url_for('submit'))
            conn.close()
    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        login_id = request.form.get('login_id', '').strip()
        password = request.form.get('password', '')
        conn = get_db()
        row = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (login_id, login_id.lower())
        ).fetchone()
        conn.close()
        if row and check_password_hash(row['password_hash'], password):
            user = User(row['id'], row['username'], row['email'],
                        row['password_hash'], row['created_at'])
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('submit'))
        error = 'invalid'
    return render_template('login.html', error=error)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/account')
@login_required
def account():
    conn = get_db()
    certs = conn.execute('''
        SELECT * FROM certifications WHERE user_id = ?
        ORDER BY created_at DESC
    ''', (current_user.id,)).fetchall()
    conn.close()
    return render_template('account.html',
                           certs=[dict(c) for c in certs])


# ─── Routes : site ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit')
@login_required
def submit():
    return render_template('submit.html')


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('submit.html', error='no_file')
    file = request.files['file']
    if not allowed_file(file.filename):
        return render_template('submit.html', error='bad_format')

    # Le nom du créateur vient du compte connecté
    creator_name = current_user.username

    filename    = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(upload_path)
    file_type = 'video' if is_video(filename) else 'image'

    # ── Analyse Sightengine (NE PAS MODIFIER) ────────────────────────────────
    ai_score = None
    if SE_SECRET:
        if file_type == 'video':
            frame = extract_video_frame(upload_path)
            if frame:
                ai_score = analyze_with_sightengine(frame)
                try:
                    os.remove(frame)
                except Exception:
                    pass
        else:
            ai_score = analyze_with_sightengine(upload_path)

    cert_id   = generate_unique_id()
    certified = ai_score is None or ai_score <= AI_THRESHOLD

    if not certified:
        _save_cert(cert_id, filename, file_type, ai_score, False,
                   f"Contenu potentiellement généré par IA (score : {ai_score:.0%})",
                   creator_name=creator_name, user_id=current_user.id)
        try:
            os.remove(upload_path)
        except Exception:
            pass
        return render_template('result.html',
                               certified=False, cert_id=cert_id,
                               ai_score=ai_score, filename=filename,
                               file_type=file_type, creator_name=creator_name)

    ext = filename.rsplit('.', 1)[1].lower()
    certified_filename = f"trueshot_{cert_id}.{ext}"
    certified_path = os.path.join(app.config['CERTIFIED_FOLDER'], certified_filename)

    if file_type == 'image':
        try:
            overlay_badge_on_image(upload_path, cert_id, certified_path)
        except Exception as e:
            print(f"[ImageOverlay] {e}")
            shutil.copy(upload_path, certified_path)
    else:
        if not overlay_badge_on_video(upload_path, cert_id, certified_path):
            shutil.copy(upload_path, certified_path)

    qr_filename = f"qr_{cert_id}.png"
    generate_qr(cert_id, request.host_url).save(
        os.path.join(app.config['QR_FOLDER'], qr_filename))

    _save_cert(cert_id, filename, file_type, ai_score, True,
               certified_file=certified_filename, qr_file=qr_filename,
               creator_name=creator_name, user_id=current_user.id)

    try:
        os.remove(upload_path)
    except Exception:
        pass

    return render_template('result.html',
                           certified=True, cert_id=cert_id,
                           ai_score=ai_score, filename=filename,
                           file_type=file_type,
                           certified_file=certified_filename,
                           qr_file=qr_filename,
                           creator_name=creator_name)


@app.route('/result/<cert_id>')
def result(cert_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM certifications WHERE id = ?', (cert_id,)).fetchone()
    conn.close()
    if not row:
        return redirect(url_for('index'))
    c = dict(row)
    return render_template('result.html',
                           certified=bool(c['certified']),
                           cert_id=cert_id,
                           ai_score=c['ai_score'],
                           filename=c['original_filename'],
                           file_type=c['file_type'],
                           certified_file=c.get('certified_file'),
                           qr_file=c.get('qr_code_file'),
                           creator_name=c.get('creator_name', ''))


@app.route('/verify/<cert_id>')
def verify(cert_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM certifications WHERE id = ?', (cert_id,)).fetchone()
    conn.close()
    if not row:
        return render_template('verify.html', found=False, cert_id=cert_id)
    cert = dict(row)
    if not cert.get('creator_name'):
        cert['creator_name'] = ''
    return render_template('verify.html', found=True, cert=cert)


@app.route('/download/<cert_id>')
def download(cert_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM certifications WHERE id = ?', (cert_id,)).fetchone()
    conn.close()
    if not row or not row['certified_file']:
        abort(404)
    path = os.path.join(app.config['CERTIFIED_FOLDER'], row['certified_file'])
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=f"TrueShot_{cert_id}_{row['original_filename']}")


@app.route('/download-qr/<cert_id>')
def download_qr(cert_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM certifications WHERE id = ?', (cert_id,)).fetchone()
    conn.close()
    if not row or not row['qr_code_file']:
        abort(404)
    path = os.path.join(app.config['QR_FOLDER'], row['qr_code_file'])
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=f"TrueShot_QR_{cert_id}.png")


# ─── Helper interne ───────────────────────────────────────────────────────────
def _save_cert(cert_id, filename, file_type, ai_score, certified,
               rejection_reason=None, certified_file=None, qr_file=None,
               creator_name=None, user_id=None):
    conn = get_db()
    conn.execute('''
        INSERT INTO certifications
            (id, original_filename, file_type, ai_score, certified,
             rejection_reason, certified_file, qr_code_file, created_at,
             creator_name, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (cert_id, filename, file_type, ai_score, int(certified),
          rejection_reason, certified_file, qr_file,
          datetime.now().strftime('%d/%m/%Y à %H:%M'),
          creator_name, user_id))
    conn.commit()
    conn.close()


# ─── Lancement ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    # Génère le badge d'aperçu pour la page d'accueil
    preview_path = os.path.join('static', 'badge_preview.png')
    if not os.path.exists(preview_path):
        try:
            create_badge('preview').save(preview_path)
        except Exception as e:
            print(f"[BadgePreview] {e}")
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
