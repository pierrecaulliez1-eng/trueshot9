import os
import re
import uuid
import smtplib
import ssl
import psycopg2
import psycopg2.extras
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE id = %s', (user_id,))
    row = dict_fetchone(cur)
    cur.close()
    conn.close()
    if row:
        return User(row['id'], row['username'], row['email'],
                    row['password_hash'], row['created_at'])
    return None


# ─── Base de données ──────────────────────────────────────────────────────────
def get_db():
    database_url = os.environ.get('DATABASE_URL', '')
    if not database_url:
        raise RuntimeError('DATABASE_URL is not set. Please configure the environment variable before connecting to the database.')
    conn = psycopg2.connect(database_url)
    return conn

def dict_fetchone(cursor):
    """Return a single row as a dict, or None."""
    row = cursor.fetchone()
    if row is None:
        return None
    cols = [desc[0] for desc in cursor.description]
    return dict(zip(cols, row))

def dict_fetchall(cursor):
    """Return all rows as a list of dicts."""
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]

def init_db():
    conn = get_db()
    cur = conn.cursor()
    # Table utilisateurs
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            SERIAL PRIMARY KEY,
            username      TEXT NOT NULL UNIQUE,
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TEXT NOT NULL
        )
    ''')
    # Table certifications
    cur.execute('''
        CREATE TABLE IF NOT EXISTS certifications (
            id                TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            file_type         TEXT NOT NULL,
            ai_score          REAL,
            certified         INTEGER NOT NULL DEFAULT 0,
            rejection_reason  TEXT,
            certified_file    TEXT,
            qr_code_file      TEXT,
            created_at        TEXT NOT NULL,
            creator_name      TEXT,
            user_id           INTEGER
        )
    ''')
    for col_def in ['creator_name TEXT', 'user_id INTEGER']:
        try:
            cur.execute(f'ALTER TABLE certifications ADD COLUMN {col_def}')
        except Exception:
            conn.rollback()
    conn.commit()
    cur.close()
    conn.close()


# ─── Email ────────────────────────────────────────────────────────────────────
def send_email(to_email, subject, body):
    mail_user     = os.environ.get('MAIL_USER', '')
    mail_password = os.environ.get('MAIL_PASSWORD', '')
    if not mail_user or not mail_password:
        print('[Email] MAIL_USER or MAIL_PASSWORD not set — skipping.')
        return
    try:
        msg = MIMEMultipart()
        msg['From']    = mail_user
        msg['To']      = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(mail_user, mail_password)
            server.sendmail(mail_user, to_email, msg.as_string())
        print(f'[Email] Sent "{subject}" to {to_email}')
    except Exception as e:
        print(f'[Email] Failed to send to {to_email}: {e}')


# ─── Utilitaires ──────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video(filename):
    return filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

def generate_unique_id():
    conn = get_db()
    cur = conn.cursor()
    for _ in range(100):
        cert_id = str(uuid.uuid4().int)[:6]
        cur.execute('SELECT 1 FROM certifications WHERE id = %s', (cert_id,))
        if not cur.fetchone():
            cur.close()
            conn.close()
            return cert_id
    cur.close()
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
    from PIL import ImageFilter
    import math
    S = 600
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    cx, cy = S // 2, S // 2
    R = 285
    shadow = Image.new('RGBA', (S, S), (0,0,0,0))
    ImageDraw.Draw(shadow).ellipse([cx-R+8, cy-R+8, cx+R+8, cy+R+8], fill=(0,0,0,100))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    img.paste(shadow, (0,0), shadow)
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx-R, cy-R, cx+R, cy+R], fill=(12, 14, 20, 255))
    draw.ellipse([cx-R, cy-R, cx+R, cy+R], outline=(74, 222, 128, 255), width=10)
    draw.ellipse([cx-R+18, cy-R+18, cx+R-18, cy+R-18], outline=(74, 222, 128, 50), width=2)
    lw = 32
    green = (74, 222, 128, 255)
    def draw_thick_line(draw, x1, y1, x2, y2, width, color):
        angle = math.atan2(y2 - y1, x2 - x1)
        dx = width / 2 * math.sin(angle)
        dy = width / 2 * math.cos(angle)
        draw.polygon([(x1-dx, y1+dy),(x1+dx, y1-dy),(x2+dx, y2-dy),(x2-dx, y2+dy)], fill=color)
    draw_thick_line(draw, cx-120, cy+10, cx-10, cy+120, lw, green)
    draw_thick_line(draw, cx-10, cy+120, cx+140, cy-110, lw, green)
    for x, y in [(cx-120, cy+10), (cx-10, cy+120), (cx+140, cy-110)]:
        r = lw // 2
        draw.ellipse([x-r, y-r, x+r, y+r], fill=green)
    return img

def overlay_badge_on_image(src, cert_id, dst):
    img = Image.open(src).convert('RGBA')
    badge = create_badge(cert_id)
    # Badge = 12% du petit côté de la photo (net à toute taille)
    target = max(60, min(img.width, img.height) // 12)
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
        target = max(60, min(video.w, video.h) // 12)
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
        elif not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            error = 'invalid_email'
        else:
            conn = get_db()
            cur = conn.cursor()
            cur.execute('SELECT 1 FROM users WHERE username = %s', (username,))
            if cur.fetchone():
                error = 'username_taken'
            else:
                cur.execute('SELECT 1 FROM users WHERE email = %s', (email,))
                if cur.fetchone():
                    error = 'email_taken'
                else:
                    cur.execute('''
                        INSERT INTO users (username, email, password_hash, created_at)
                        VALUES (%s, %s, %s, %s)
                    ''', (username, email, generate_password_hash(password),
                          datetime.now().strftime('%d/%m/%Y à %H:%M')))
                    conn.commit()
                    cur.execute('SELECT * FROM users WHERE username = %s', (username,))
                    row = dict_fetchone(cur)
                    cur.close()
                    conn.close()
                    user = User(row['id'], row['username'], row['email'],
                                row['password_hash'], row['created_at'])
                    login_user(user)
                    try:
                        send_email(
                            email,
                            'Bienvenue sur TrueShot',
                            f'Bonjour {username},\n\nBienvenue sur TrueShot ! Votre compte a été créé avec succès.\n\nBonne certification !\nL\'équipe TrueShot'
                        )
                        send_email(
                            'muguet.marcq@gmail.com',
                            'Nouveau compte TrueShot',
                            f'Un nouveau compte vient d\'être créé.\n\nNom d\'utilisateur : {username}\nAdresse email : {email}'
                        )
                    except Exception as e:
                        print(f'[Email] Unexpected error during registration emails: {e}')
                    return redirect(url_for('submit'))
            cur.close()
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
        cur = conn.cursor()
        cur.execute(
            'SELECT * FROM users WHERE username = %s OR email = %s',
            (login_id, login_id.lower())
        )
        row = dict_fetchone(cur)
        cur.close()
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
    cur = conn.cursor()
    cur.execute('''
        SELECT * FROM certifications WHERE user_id = %s
        ORDER BY created_at DESC
    ''', (current_user.id,))
    certs = dict_fetchall(cur)
    cur.close()
    conn.close()
    return render_template('account.html', certs=certs)


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
    cur = conn.cursor()
    cur.execute('SELECT * FROM certifications WHERE id = %s', (cert_id,))
    row = dict_fetchone(cur)
    cur.close()
    conn.close()
    if not row:
        return redirect(url_for('index'))
    return render_template('result.html',
                           certified=bool(row['certified']),
                           cert_id=cert_id,
                           ai_score=row['ai_score'],
                           filename=row['original_filename'],
                           file_type=row['file_type'],
                           certified_file=row.get('certified_file'),
                           qr_file=row.get('qr_code_file'),
                           creator_name=row.get('creator_name', ''))


@app.route('/verify/<cert_id>')
def verify(cert_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM certifications WHERE id = %s', (cert_id,))
    row = dict_fetchone(cur)
    cur.close()
    conn.close()
    if not row:
        return render_template('verify.html', found=False, cert_id=cert_id)
    if not row.get('creator_name'):
        row['creator_name'] = ''
    return render_template('verify.html', found=True, cert=row)


@app.route('/download/<cert_id>')
def download(cert_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM certifications WHERE id = %s', (cert_id,))
    row = dict_fetchone(cur)
    cur.close()
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
    cur = conn.cursor()
    cur.execute('SELECT * FROM certifications WHERE id = %s', (cert_id,))
    row = dict_fetchone(cur)
    cur.close()
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
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO certifications
            (id, original_filename, file_type, ai_score, certified,
             rejection_reason, certified_file, qr_code_file, created_at,
             creator_name, user_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (cert_id, filename, file_type, ai_score, int(certified),
          rejection_reason, certified_file, qr_file,
          datetime.now().strftime('%d/%m/%Y à %H:%M'),
          creator_name, user_id))
    conn.commit()
    cur.close()
    conn.close()


# ─── Lancement ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        init_db()
    except Exception as e:
        print(f'[DB] Warning: init_db() failed at startup: {e}')
    # Génère le badge d'aperçu pour la page d'accueil
    preview_path = os.path.join('static', 'badge_preview.png')
    if not os.path.exists(preview_path):
        try:
            create_badge('preview').save(preview_path)
        except Exception as e:
            print(f"[BadgePreview] {e}")
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
