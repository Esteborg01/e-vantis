import os
import time
import json
import re
import sqlite3
import uuid
import unicodedata
import hashlib
import hmac
import secrets
import stripe
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, Depends, HTTPException, Header, Response, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from pydantic import BaseModel, Field, StrictBool
from passlib.context import CryptContext
from passlib.exc import UnknownHashError
from dotenv import load_dotenv
from openai import OpenAI

from routes_curriculum import router as curriculum_router


# ----------------------------
# Quotas por plan / módulo (FASE 8)
# ----------------------------
QUOTAS = {
    "free": {"lesson": 10, "exam": 5, "enarm": 0, "gpc_summary": 0},
    "pro": {"lesson": 250, "exam": 150, "enarm": 100, "gpc_summary": 50},
    "premium": {"lesson": 1000, "exam": 500, "enarm": 300, "gpc_summary": 300},
}


def _yyyymm_utc() -> str:
    return datetime.utcnow().strftime("%Y%m")


def _quota_limit(plan: str, module: str) -> int:
    plan = (plan or "free").strip().lower()
    module = (module or "").strip().lower()

    if plan not in QUOTAS:
        plan = "free"
    return int(QUOTAS.get(plan, {}).get(module, 0) or 0)


def usage_monthly_get_count(conn, user_id: str, module: str, yyyymm: str) -> int:
    cur = conn.execute(
        """
        SELECT count FROM usage_monthly
        WHERE user_id = ? AND module = ? AND yyyymm = ?
        """,
        (user_id, module, yyyymm),
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def usage_monthly_increment(conn, user_id: str, module: str, yyyymm: str) -> None:
    retries = 6
    backoff = 0.05  # 50ms

    for attempt in range(retries):
        try:
            conn.execute(
                """
                INSERT INTO usage_monthly (user_id, module, yyyymm, count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(user_id, module, yyyymm)
                DO UPDATE SET count = count + 1
                """,
                (user_id, module, yyyymm),
            )
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise


def _rate_limit_key(user_id: str, ip: str, endpoint: str) -> str:
    return f"{endpoint}:{user_id}:{ip}"


def enforce_rate_limit(conn, user_id: str, ip: str, endpoint: str, limit_per_minute: int = 30):
    """
    Rate limit por minuto usando SQLite.
    Estrategia anti-lock:
    - BEGIN IMMEDIATE (toma el lock de escritura rápido)
    - transacción MUY corta
    - retry con backoff si DB está locked
    """
    key = _rate_limit_key(user_id, ip, endpoint)
    window = int(time.time() // 60)  # minuto actual
    retries = 6
    backoff = 0.05  # 50ms

    for attempt in range(retries):
        try:
            conn.execute("BEGIN IMMEDIATE;")

            row = conn.execute(
                "SELECT window_start, count FROM rate_limit WHERE key=?",
                (key,),
            ).fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO rate_limit (key, window_start, count) VALUES (?, ?, 1)",
                    (key, window),
                )
                conn.execute("COMMIT;")
                return

            window_start, count = int(row[0]), int(row[1])

            if window_start != window:
                conn.execute(
                    "UPDATE rate_limit SET window_start=?, count=1 WHERE key=?",
                    (window, key),
                )
                conn.execute("COMMIT;")
                return

            if count >= int(limit_per_minute):
                conn.execute("ROLLBACK;")
                raise HTTPException(status_code=429, detail="Rate limit excedido. Intenta de nuevo en 1 minuto.")

            conn.execute(
                "UPDATE rate_limit SET count=count+1 WHERE key=?",
                (key,),
            )
            conn.execute("COMMIT;")
            return

        except sqlite3.OperationalError as e:
            # liberamos la transacción si quedó abierta
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass

            if "locked" in str(e).lower() and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise


def enforce_idempotency(conn, user_id: str, idem_key: str, ttl_seconds: int = 30):
    """
    Idempotency key anti-lock:
    - transacción corta (BEGIN IMMEDIATE)
    - retry/backoff si SQLite está locked
    - commit inmediato
    """
    now = int(time.time())
    retries = 6
    backoff = 0.05  # 50ms

    for attempt in range(retries):
        try:
            conn.execute("BEGIN IMMEDIATE;")

            # Limpieza TTL (write corta)
            conn.execute(
                "DELETE FROM idempotency_keys WHERE created_at < ?",
                (now - ttl_seconds,),
            )

            # Check existencia
            row = conn.execute(
                "SELECT 1 FROM idempotency_keys WHERE user_id=? AND key=?",
                (user_id, idem_key),
            ).fetchone()

            if row:
                conn.execute("ROLLBACK;")
                raise HTTPException(
                    status_code=409,
                    detail="Solicitud duplicada (idempotency key repetida).",
                )

            # Registrar key
            conn.execute(
                "INSERT INTO idempotency_keys (user_id, key, created_at) VALUES (?, ?, ?)",
                (user_id, idem_key, now),
            )

            conn.execute("COMMIT;")
            return

        except sqlite3.OperationalError as e:
            # rollback defensivo
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass

            if "locked" in str(e).lower() and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

app = FastAPI(title="E-VANTIS")

# =========================
# CORS (DEBE IR AQUÍ ARRIBA)
# =========================
FRONTEND_BASE_URL = (os.getenv("FRONTEND_BASE_URL", "") or "").strip().rstrip("/")

default_origins = [
    "https://evantis-frontend.onrender.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
if FRONTEND_BASE_URL:
    default_origins.append(FRONTEND_BASE_URL)

env_origins = (os.getenv("EVANTIS_CORS_ORIGINS", "") or "").strip()
if env_origins:
    allowed = [o.strip().rstrip("/") for o in env_origins.split(",") if o.strip()]
else:
    allowed = [o.rstrip("/") for o in default_origins]

# Dedup
allowed = sorted({o for o in allowed if o})

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,              # ✅ ESTA es la lista correcta
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],                # ✅ incluye Authorization/X-API-Key/Idempotency-Key
    expose_headers=["*"],
)

print(">>> CORS allowed =", allowed)

print(">>> LOADED app.py FROM:", __file__)
app.include_router(curriculum_router)

bearer_scheme = HTTPBearer(auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# STRIPE CONFIG (GLOBAL)
# =========================
STRIPE_MODE = os.getenv("STRIPE_MODE", "test").strip().lower()  # "test" o "live"
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

STRIPE_PRICE_PRO = os.getenv("STRIPE_PRICE_PRO", "")
STRIPE_PRICE_PREMIUM = os.getenv("STRIPE_PRICE_PREMIUM", "")

FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "").strip().rstrip("/")

def stripe_price_for_plan(plan: str) -> str:
    plan = (plan or "").strip().lower()
    if plan == "pro":
        return STRIPE_PRICE_PRO
    if plan == "premium":
        return STRIPE_PRICE_PREMIUM
    raise ValueError("Plan inválido (pro|premium)")

# Set API key early if provided (startup también lo refuerza)
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

def assert_stripe_mode():
    if STRIPE_MODE not in ("test", "live"):
        raise RuntimeError("STRIPE_MODE debe ser 'test' o 'live'")

    if STRIPE_SECRET_KEY:
        if STRIPE_MODE == "test" and not STRIPE_SECRET_KEY.startswith("sk_test_"):
            raise RuntimeError("Anti-mezcla: STRIPE_MODE=test pero STRIPE_SECRET_KEY no es sk_test_")
        if STRIPE_MODE == "live" and not STRIPE_SECRET_KEY.startswith("sk_live_"):
            raise RuntimeError("Anti-mezcla: STRIPE_MODE=live pero STRIPE_SECRET_KEY no es sk_live_")

    # Si vas a usar webhook, exige whsec_
    if STRIPE_WEBHOOK_SECRET and not STRIPE_WEBHOOK_SECRET.startswith("whsec_"):
        raise RuntimeError("STRIPE_WEBHOOK_SECRET inválido")

    # Si tienes webhook secret, exige price IDs
    if STRIPE_WEBHOOK_SECRET:
        if not STRIPE_PRICE_PRO or not STRIPE_PRICE_PREMIUM:
            raise RuntimeError("Faltan STRIPE_PRICE_PRO / STRIPE_PRICE_PREMIUM")

def assert_stripe_ready():
    """
    Reglas mínimas para usar Checkout/Portal:
    - STRIPE_SECRET_KEY presente y consistente con STRIPE_MODE
    - Price IDs presentes
    """
    assert_stripe_mode()

    if not STRIPE_SECRET_KEY:
        raise RuntimeError("Falta STRIPE_SECRET_KEY (Stripe server-side no puede operar).")

    if not STRIPE_PRICE_PRO or not STRIPE_PRICE_PREMIUM:
        raise RuntimeError("Faltan STRIPE_PRICE_PRO / STRIPE_PRICE_PREMIUM (requeridos para checkout).")

# ----------------------------
# Email verification config (A3)
# ----------------------------
# A3 QA MODE
# Permite login aunque el correo no esté verificado.
# ⚠️ SOLO PARA QA / DESARROLLO. En producción debe ser 0.

EVANTIS_EMAIL_VERIFY_ENABLED = os.getenv("EVANTIS_EMAIL_VERIFY_ENABLED", "1") == "1"
EVANTIS_EMAIL_VERIFY_TTL_SECONDS = int(os.getenv("EVANTIS_EMAIL_VERIFY_TTL_SECONDS", "86400"))  # 24h
EVANTIS_EMAIL_FROM = os.getenv("EVANTIS_EMAIL_FROM", "no-reply@evantis.local")

# Si NO tienes proveedor de correo, deja SMTP_HOST vacío y se imprimirá el link en logs.
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "1") == "1"

# URL base para verificación: frontend o backend. Ej:
# https://evantis-frontend.onrender.com/verify-email
# o https://e-vantis-api.onrender.com/auth/verify-email (si lo manejas en backend)

EVANTIS_APP_URL = os.getenv(
    "EVANTIS_APP_URL",
    "https://evantis-frontend.onrender.com"
)

EVANTIS_EMAIL_VERIFY_BASE_URL = os.getenv(
    "EVANTIS_EMAIL_VERIFY_BASE_URL",
    EVANTIS_APP_URL.rstrip("/") + "/verify-email"
)

# Si =1, /auth/register devuelve verify_link (útil para QA). En prod déjalo en 0.
EVANTIS_RETURN_VERIFY_LINK = os.getenv("EVANTIS_RETURN_VERIFY_LINK", "0") == "1"

# Si =1, permite login aunque el correo no esté verificado (solo QA).
# En prod real: dejar en 0.
EVANTIS_ALLOW_UNVERIFIED_LOGIN = os.getenv("EVANTIS_ALLOW_UNVERIFIED_LOGIN", "0") == "1"

# ----------------------------
# SQLite config (define BEFORE functions that use DB_PATH)
# ----------------------------
DB_PATH = os.getenv("EVANTIS_DB_PATH", str(BASE_DIR / "evantis.sqlite3"))
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

PRO_GPC_SUMMARY_MONTHLY_CAP = int(os.getenv("PRO_GPC_SUMMARY_MONTHLY_CAP", "30"))
PREMIUM_GPC_SUMMARY_MONTHLY_CAP = int(os.getenv("PREMIUM_GPC_SUMMARY_MONTHLY_CAP", "200"))

# ----------------------------
# Plans, study modes, modules
# ----------------------------
Plan = Literal["free", "pro", "premium"]
Level = Literal["auto", "pregrado", "internado"]
StudyMode = Literal["basico", "clinico", "internado", "examen"]
Module = Literal["lesson", "exam", "enarm", "gpc_summary"]


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def assert_config_or_die() -> None:
    required = ["EVANTIS_JWT_SECRET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")

    secret = os.getenv("EVANTIS_JWT_SECRET", "")
    if len(secret) < 16:
        raise RuntimeError("EVANTIS_JWT_SECRET demasiado corto (mínimo 16 caracteres).")


# ----------------------------
# JWT auth + password hashing
# ----------------------------
JWT_SECRET = os.getenv("EVANTIS_JWT_SECRET", "")
if not JWT_SECRET or len(JWT_SECRET) < 16:
    raise RuntimeError("EVANTIS_JWT_SECRET no configurado o demasiado corto (mínimo 16 chars).")

JWT_ALG = "HS256"
JWT_EXPIRE_MIN = int(os.getenv("EVANTIS_JWT_EXPIRE_MIN", "43200"))

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")


def hash_password(p: str) -> str:
    return pwd_context.hash(p)


def verify_password(p: str, ph: str) -> bool:
    if not ph or not isinstance(ph, str):
        return False
    try:
        return pwd_context.verify(p, ph)
    except UnknownHashError:
        return False
    except Exception:
        return False


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def is_sha256_hex(s: str) -> bool:
    if not s or not isinstance(s, str):
        return False
    if len(s) != 64:
        return False
    return all(c in "0123456789abcdef" for c in s.lower())


def verify_password_sha256(plain: str, stored_hex: str) -> bool:
    if not is_sha256_hex(stored_hex):
        return False
    computed = sha256_hex(plain)
    return hmac.compare_digest(computed, stored_hex)


def create_access_token(user_id: str, plan: Plan, sid: str, jti: str) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "plan": plan,
        "sid": sid,
        "jti": jti,
        "iat": now,
        "exp": now + (JWT_EXPIRE_MIN * 60),
    }
    import jwt
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_access_token(token: str) -> dict:
    import jwt
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Falta token Bearer.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)


# ----------------------------
# DB helpers
# ----------------------------
def db_set_email_verification(user_id: str, token: str, expires_at: int) -> None:
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET email_verify_token=?, email_verify_expires_at=?, email_verified=0 WHERE user_id=?",
            (token, int(expires_at), user_id),
        )
        conn.commit()

def db_get_user_by_verify_token(token: str):
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT user_id, email, email_verify_expires_at, email_verified FROM users WHERE email_verify_token=?",
            (token,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "user_id": row[0],
            "email": row[1],
            "expires_at": int(row[2] or 0),
            "email_verified": bool(int(row[3] or 0)),
        }

def db_register_stripe_event(event_id: str) -> bool:
    """
    True  -> evento nuevo
    False -> evento duplicado (ya procesado)
    """
    if not event_id:
        return True

    with db_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO stripe_events (event_id, created_at) VALUES (?, ?)",
                (event_id, int(time.time()))
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def plan_from_price_id(price_id: str) -> str:
    if price_id == STRIPE_PRICE_PREMIUM:
        return "premium"
    if price_id == STRIPE_PRICE_PRO:
        return "pro"
    return "free"

def db_mark_email_verified(user_id: str) -> None:
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET email_verified=1, email_verify_token=NULL, email_verify_expires_at=NULL WHERE user_id=?",
            (user_id,),
        )
        conn.commit()

def db_set_stripe_customer(user_id: str, customer_id: str) -> None:
    if not customer_id:
        return
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET stripe_customer_id=? WHERE user_id=?",
            (customer_id, user_id),
        )
        try:
            conn.commit()
        except Exception:
            pass


def db_set_stripe_subscription(user_id: str, subscription_id: str, status: str | None = None) -> None:
    if not subscription_id:
        return
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET stripe_subscription_id=?, stripe_status=COALESCE(?, stripe_status) WHERE user_id=?",
            (subscription_id, status, user_id),
        )
        try:
            conn.commit()
        except Exception:
            pass


def db_apply_plan_from_stripe(user_id: str, plan: str, is_active: bool, status: str | None = None) -> None:
    plan = (plan or "free").strip().lower()
    if plan not in ("free", "pro", "premium"):
        plan = "free"
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET plan=?, is_active=?, stripe_status=COALESCE(?, stripe_status) WHERE user_id=?",
            (plan, 1 if is_active else 0, status, user_id),
        )
        try:
            conn.commit()
        except Exception:
            pass

def db_get_user_by_id(user_id: str) -> dict | None:
    with db_conn() as conn:
        cur = conn.execute(
            """
            SELECT
                user_id,
                email,
                plan,
                is_active,
                email_verified,
                stripe_customer_id,
                stripe_subscription_id,
                stripe_status
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        # ✅ Soporta tanto sqlite3.Row como tuplas
        if isinstance(row, sqlite3.Row):
            return dict(row)

        cols = [d[0] for d in (cur.description or [])]
        return dict(zip(cols, row))

def _make_verify_link(token: str) -> str:
    base = (EVANTIS_EMAIL_VERIFY_BASE_URL or "").strip().rstrip("/")
    # El frontend puede recibir ?token=... y llamar a backend /auth/verify-email si quieres.
    return f"{base}?token={token}"


def send_verify_email(email: str, token: str) -> str:
    """
    En producción: envía correo si hay SMTP.
    Sin proveedor: imprime el link en logs y regresa el link.
    """
    link = _make_verify_link(token)

    # Fallback: sin SMTP → imprimir link (MVP)
    if not SMTP_HOST:
        print(f"[EMAIL_VERIFY_LINK] email={email} link={link}")
        return link

    # SMTP simple (opcional)
    try:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = "Verifica tu correo — E-Vantis"
        msg["From"] = EVANTIS_EMAIL_FROM
        msg["To"] = email
        msg.set_content(
            "Verifica tu correo para activar tu cuenta.\n\n"
            f"Link de verificación:\n{link}\n\n"
            "Si no solicitaste esto, ignora este correo."
        )

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            if SMTP_USE_TLS:
                server.starttls()
            if SMTP_USER and SMTP_PASS:
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        return link
    except Exception as e:
        # Si falla SMTP, no bloquees MVP: imprime link y listo
        print(f"[EMAIL_VERIFY_FALLBACK] smtp_failed={repr(e)} email={email} link={link}")
        return link

def db_conn():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row  # ✅ CLAVE: permite dict(row) y acceso por nombre

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def db_init():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            history_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS stripe_events (
            event_id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS teach_cache (
            cache_key TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            subject TEXT NOT NULL,
            level TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL,
            title TEXT NOT NULL,
            lesson TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            api_key TEXT PRIMARY KEY,
            role TEXT NOT NULL,          -- 'student' o 'admin'
            label TEXT NOT NULL,
            is_active INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_used_at TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            plan TEXT NOT NULL DEFAULT 'free',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            last_login_at TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            user_id TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            subject_id TEXT,
            topic_id TEXT,
            module TEXT,
            used_web_search INTEGER NOT NULL DEFAULT 0,
            model TEXT,
            approx_output_chars INTEGER NOT NULL DEFAULT 0
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_monthly (
            user_id TEXT NOT NULL,
            yyyymm TEXT NOT NULL,
            module TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, yyyymm, module)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS rate_limit (
            key TEXT PRIMARY KEY,
            window_start INTEGER NOT NULL,
            count INTEGER NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, key)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            sid TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            jti TEXT NOT NULL UNIQUE,
            issued_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            ip TEXT,
            user_agent TEXT,
            last_seen_at INTEGER
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_jti ON user_sessions(jti)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(user_id, is_active)")
        conn.commit()

        # ----------------------------
        # Email verification (MVP)
        # ----------------------------
        try:
            conn.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN email_verify_token TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE users ADD COLUMN email_verify_expires_at INTEGER")
        except Exception:
            pass

        # Backfill defensivo para bases existentes
        try:
            conn.execute("UPDATE users SET email_verified=0 WHERE email_verified IS NULL")
        except Exception:
            pass


        # ----------------------------
        # Chat threads + messages
        # ----------------------------
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            subject_id TEXT,
            topic_id TEXT,
            lesson_session_id TEXT,
            title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        conn.commit()


def db_log_usage(
    user_id: str,
    endpoint: str,
    subject_id: Optional[str],
    topic_id: Optional[str],
    module: Optional[str],
    used_web_search: bool,
    model: str,
    approx_output_chars: int,
):
    retries = 6
    backoff = 0.05  # 50ms

    for attempt in range(retries):
        try:
            with db_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO usage_log(
                        created_at, user_id, endpoint, subject_id, topic_id,
                        module, used_web_search, model, approx_output_chars
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _now_iso(),
                        user_id,
                        endpoint,
                        subject_id,
                        topic_id,
                        module,
                        1 if used_web_search else 0,
                        model,
                        int(approx_output_chars or 0),
                    ),
                )
                try:
                    conn.commit()
                except Exception:
                    pass
            return

        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue

            if os.getenv("EVANTIS_LOG_DB_ERRORS", "0") == "1":
                try:
                    import logging
                    logging.exception("db_log_usage failed: %s", e)
                except Exception:
                    print("db_log_usage failed:", repr(e))
            return

        except Exception as e:
            if os.getenv("EVANTIS_LOG_DB_ERRORS", "0") == "1":
                try:
                    import logging
                    logging.exception("db_log_usage failed: %s", e)
                except Exception:
                    print("db_log_usage failed:", repr(e))
            return

def db_get_user_by_email(email: str):
    with db_conn() as conn:
        cur = conn.execute(
            """
            SELECT
              user_id,
              email,
              password_hash,
              plan,
              is_active,
              email_verified,
              email_verify_expires_at,
              email_verify_token
            FROM users
            WHERE email = ?
            """,
            (email.strip().lower(),),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "user_id": row[0],
            "email": row[1],
            "password_hash": row[2],
            "plan": row[3],
            "is_active": bool(int(row[4] or 0)),
            "email_verified": bool(int(row[5] or 0)),
            "email_verify_expires_at": int(row[6] or 0),
            "email_verify_token": row[7],
        }


def db_create_user(email: str, password_hash: str, plan: Plan = "free"):
    user_id = "u_" + uuid.uuid4().hex
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO users(user_id, email, password_hash, plan, is_active, created_at)
            VALUES(?, ?, ?, ?, 1, ?)
            """,
            (user_id, email.strip().lower(), password_hash, plan, _now_iso()),
        )
        conn.commit()
    return user_id


def db_touch_login(user_id: str):
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE user_id = ?",
            (_now_iso(), user_id),
        )
        conn.commit()


def db_update_password_hash(user_id: str, new_hash: str) -> None:
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, last_login_at = ? WHERE user_id = ?",
            (new_hash, datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()

def db_set_plan(user_id: str, plan: str) -> None:
    plan = (plan or "free").strip().lower()
    if plan not in ("free", "pro", "premium"):
        plan = "free"
    with db_conn() as conn:
        conn.execute("UPDATE users SET plan=? WHERE user_id=?", (plan, user_id))
        conn.commit()


def db_update_stripe_fields(
    user_id: str,
    customer_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
    status: Optional[str] = None,
    current_period_end: Optional[int] = None,
) -> None:
    with db_conn() as conn:
        conn.execute(
            """
            UPDATE users
            SET
              stripe_customer_id = COALESCE(?, stripe_customer_id),
              stripe_subscription_id = COALESCE(?, stripe_subscription_id),
              stripe_status = COALESCE(?, stripe_status),
              stripe_current_period_end = COALESCE(?, stripe_current_period_end)
            WHERE user_id = ?
            """,
            (customer_id, subscription_id, status, current_period_end, user_id),
        )
        conn.commit()


def db_get_user_by_stripe_customer(customer_id: str):
    with db_conn() as conn:
        row = conn.execute(
            "SELECT user_id, email, plan FROM users WHERE stripe_customer_id=?",
            (customer_id,),
        ).fetchone()
        if not row:
            return None
        return {"user_id": row[0], "email": row[1], "plan": row[2]}

def db_load_session(session_id: str):
    with db_conn() as conn:
        cur = conn.execute("SELECT summary, history_json FROM sessions WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        if not row:
            return "", []
        summary, history_json = row
        try:
            history = json.loads(history_json)
        except Exception:
            history = []
        return summary or "", history or []


def db_save_session(session_id: str, summary: str, history: List[Dict[str, str]]):
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions(session_id, summary, history_json, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                summary=excluded.summary,
                history_json=excluded.history_json,
                updated_at=excluded.updated_at
            """,
            (session_id, summary or "", json.dumps(history, ensure_ascii=False), _now_iso()),
        )
        conn.commit()


def db_delete_session(session_id: str):
    with db_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def db_revoke_user_sessions(user_id: str):
    with db_conn() as conn:
        conn.execute(
            "UPDATE user_sessions SET is_active=0 WHERE user_id=? AND is_active=1",
            (user_id,),
        )
        conn.commit()


def db_insert_session(
    user_id: str,
    sid: str,
    jti: str,
    issued_at: int,
    expires_at: int,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
):
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO user_sessions(sid, user_id, jti, issued_at, expires_at, is_active, ip, user_agent, last_seen_at)
            VALUES(?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (sid, user_id, jti, issued_at, expires_at, ip, user_agent, issued_at),
        )
        conn.commit()


def db_is_session_active(user_id: str, jti: str) -> bool:
    with db_conn() as conn:
        cur = conn.execute(
            """
            SELECT 1 FROM user_sessions
            WHERE user_id=? AND jti=? AND is_active=1 AND expires_at > ?
            """,
            (user_id, jti, int(time.time())),
        )
        return cur.fetchone() is not None


def db_touch_session_last_seen(user_id: str, jti: str):
    try:
        with db_conn() as conn:
            conn.execute(
                "UPDATE user_sessions SET last_seen_at=? WHERE user_id=? AND jti=?",
                (int(time.time()), user_id, jti),
            )
            conn.commit()
    except Exception:
        pass


# ----------------------------
# Auth: require_user / plan gating
# ----------------------------
def require_user(token: str = Depends(oauth2_scheme)) -> dict:
    data = decode_access_token(token)
    user_id = data.get("sub")
    jti = data.get("jti")

    if not user_id:
        raise HTTPException(status_code=401, detail="Token inválido (sin sub).")
    if not jti:
        raise HTTPException(status_code=401, detail="Sesión inválida o revocada.")

    u = db_get_user_by_id(user_id)
    if not u or not u["is_active"]:
        raise HTTPException(status_code=401, detail="Usuario no activo.")
    if not db_is_session_active(user_id, jti):
        raise HTTPException(status_code=401, detail="Sesión inválida o revocada.")

    db_touch_session_last_seen(user_id, jti)

    # Fuente de verdad del plan: DB
    plan_db = (u.get("plan") or "free").strip().lower()
    if plan_db not in ("free", "pro", "premium"):
        plan_db = "free"

    return {
        "user_id": user_id,
        "plan": plan_db,
        "email": (u.get("email") or "").strip().lower(),
        "stripe_customer_id": (u.get("stripe_customer_id") or None),
        "stripe_subscription_id": (u.get("stripe_subscription_id") or None),
        "stripe_status": (u.get("stripe_status") or None),
        "is_active": bool(int(u.get("is_active") or 0)),
    }

def require_plan(min_plan: Plan, user: dict = Depends(require_user)) -> dict:
    order = {"free": 0, "pro": 1, "premium": 2}
    if order.get(user["plan"], 0) < order.get(min_plan, 0):
        raise HTTPException(status_code=403, detail=f"Requiere plan {min_plan} o superior.")
    return user


def can_use_web_search(plan: Plan) -> bool:
    return plan in ("pro", "premium")


def can_use_enarm(plan: Plan) -> bool:
    return plan in ("pro", "premium")


def can_use_gpc_summary(plan: Plan) -> bool:
    return plan in ("pro", "premium")


def gpc_summary_monthly_cap(plan: str) -> int:
    p = (plan or "free").strip().lower()
    if p == "pro":
        return PRO_GPC_SUMMARY_MONTHLY_CAP
    if p == "premium":
        return PREMIUM_GPC_SUMMARY_MONTHLY_CAP
    return 0


def compute_capabilities(plan: str) -> dict:
    plan = (plan or "free").strip().lower()
    return {
        "plan": plan,
        "modules": {
            "lesson": True,
            "exam": True,
            "enarm": can_use_enarm(plan),            # Pro/Premium
            "gpc_summary": can_use_gpc_summary(plan) # Pro/Premium
        },
        "features": {
            "use_guides": can_use_web_search(plan),  # Pro/Premium
        },
        "quotas": {
            "gpc_summary_per_month": gpc_summary_monthly_cap(plan),
            "enarm_cases_per_month": 0 if plan == "free" else (40 if plan == "pro" else 120),
        },
    }


def has_review_questions(md: str) -> bool:
    if not md or not isinstance(md, str):
        return False
    tail = md[-3000:].lower()
    if "## preguntas de repaso" in tail:
        return True
    return ("preguntas" in tail and "repaso" in tail)


@app.get("/auth/me")
def auth_me(user: dict = Depends(require_user)):
    plan = (user.get("plan") or "free").strip().lower()
    return {
        "user_id": user["user_id"],
        "plan": plan,
        "capabilities": compute_capabilities(plan),
    }


@app.post("/auth/logout")
def logout(user: dict = Depends(require_user), token: str = Depends(oauth2_scheme)):
    data = decode_access_token(token)
    user_id = data.get("sub")
    jti = data.get("jti")

    if user_id and jti:
        with db_conn() as conn:
            conn.execute(
                "UPDATE user_sessions SET is_active=0 WHERE user_id=? AND jti=?",
                (user_id, jti),
            )
            conn.commit()

    return {"ok": True}


# ----------------------------
# Monthly quota (GPC summary / CHAT)
# ----------------------------
def db_get_monthly_count(user_id: str, module: str) -> int:
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT count FROM usage_monthly WHERE user_id=? AND yyyymm=? AND module=?",
            (user_id, _yyyymm_utc(), module),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def db_inc_monthly_count(user_id: str, module: str):
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO usage_monthly(user_id, yyyymm, module, count)
            VALUES(?, ?, ?, 1)
            ON CONFLICT(user_id, yyyymm, module)
            DO UPDATE SET count = count + 1
            """,
            (user_id, _yyyymm_utc(), module),
        )
        conn.commit()


def enforce_gpc_summary_quota(user_id: str, plan: str):
    cap = gpc_summary_monthly_cap(plan)
    if cap <= 0:
        raise HTTPException(status_code=403, detail="Resumen GPC disponible solo en Pro/Premium.")
    used = db_get_monthly_count(user_id, "gpc_summary")
    if used >= cap:
        raise HTTPException(
            status_code=429,
            detail=f"Límite mensual alcanzado para Resumen GPC ({used}/{cap}).",
        )


def consume_gpc_summary_quota(user_id: str):
    db_inc_monthly_count(user_id, "gpc_summary")


def chat_monthly_cap(plan: str) -> int:
    p = (plan or "free").strip().lower()
    if p == "free":
        return 50
    if p == "pro":
        return 500
    if p == "premium":
        return 2000
    return 50


def enforce_chat_quota(user_id: str, plan: str):
    cap = chat_monthly_cap(plan)
    used = db_get_monthly_count(user_id, "chat")
    if used >= cap:
        raise HTTPException(
            status_code=429,
            detail=f"Límite mensual alcanzado para Chat ({used}/{cap}).",
        )


def consume_chat_quota(user_id: str):
    db_inc_monthly_count(user_id, "chat")


# ----------------------------
# API keys (admin panel)
# ----------------------------
def db_get_key(api_key: str):
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT api_key, role, label, is_active FROM api_keys WHERE api_key = ?",
            (api_key,),
        )
        return cur.fetchone()


def db_touch_key(api_key: str):
    with db_conn() as conn:
        conn.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE api_key = ?",
            (_now_iso(), api_key),
        )
        conn.commit()


def db_create_key(api_key: str, role: str, label: str):
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO api_keys(api_key, role, label, is_active, created_at, last_used_at)
            VALUES (?, ?, ?, 1, ?, NULL)
            """,
            (api_key, role, label, _now_iso()),
        )
        conn.commit()


def db_revoke_key(api_key: str):
    with db_conn() as conn:
        conn.execute("UPDATE api_keys SET is_active = 0 WHERE api_key = ?", (api_key,))
        conn.commit()


def db_list_keys():
    with db_conn() as conn:
        cur = conn.execute(
            "SELECT api_key, role, label, is_active, created_at, last_used_at FROM api_keys ORDER BY created_at DESC"
        )
        return cur.fetchall()


def require_student_or_admin(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Falta X-API-Key")
    row = db_get_key(x_api_key)
    if not row:
        raise HTTPException(status_code=403, detail="API key inválida")
    api_key, role, label, is_active = row
    if int(is_active) != 1:
        raise HTTPException(status_code=403, detail="API key revocada")
    try:
        db_touch_key(x_api_key)
    except Exception:
        pass
    return x_api_key


def require_admin(x_api_key: str = Depends(require_student_or_admin)) -> str:
    row = db_get_key(x_api_key)
    if not row:
        raise HTTPException(status_code=403, detail="No autorizado")
    _, role, _, is_active = row
    if int(is_active) != 1 or role != "admin":
        raise HTTPException(status_code=403, detail="Requiere rol admin")
    return x_api_key

    # ----------------------------
    # Subscription lifecycle
    # ----------------------------
    if event_type in (
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    ):
        customer_id = (obj.get("customer") or "").strip()
        sub_id = (obj.get("id") or "").strip()
        status = (obj.get("status") or "").strip()
        current_period_end = int(obj.get("current_period_end") or 0)

        # price id (primera línea del subscription item)
        price_id = ""
        try:
            items = (obj.get("items") or {}).get("data") or []
            if items:
                price = (items[0].get("price") or {})
                price_id = (price.get("id") or "").strip()
        except Exception:
            price_id = ""

        # Resolver usuario por customer_id
        user = db_get_user_by_stripe_customer(customer_id) if customer_id else None
        if not user and customer_id:
            try:
                cust = stripe.Customer.retrieve(customer_id)
                meta = (cust.get("metadata") or {})
                uid = (meta.get("evantis_user_id") or "").strip()
                if uid:
                    # linkear en DB y continuar
                    db_update_stripe_fields(user_id=uid, customer_id=customer_id)
                    user = db_get_user_by_id(uid)
            except Exception as e:
                print("[STRIPE_WEBHOOK] customer_retrieve_failed:", repr(e))

        if not user:
            print(f"[STRIPE_WEBHOOK] unlinked_customer event={event_type} customer={customer_id}")
            return {"ok": True, "unlinked_customer": True}


        # Persistir estado Stripe (fuente de verdad)
        db_update_stripe_fields(
            user_id=user["user_id"],
            customer_id=customer_id,
            subscription_id=sub_id,
            status=status,
            current_period_end=current_period_end if current_period_end else None,
        )

        # Downgrade NO agresivo: solo terminales
        terminal = status in ("canceled", "incomplete_expired")
        if event_type == "customer.subscription.deleted" or terminal:
            db_set_plan(user["user_id"], "free")
            return {"ok": True, "plan": "free", "status": status}

        # Activo o trial: asignar plan según price
        if status in ("active", "trialing"):
            plan = plan_from_price_id(price_id)
            db_set_plan(user["user_id"], plan)
            return {"ok": True, "plan": plan, "status": status}

        # Estados intermedios (past_due/unpaid/incomplete/etc):
        # NO degradamos aquí; solo registramos status (ya guardado).
        return {"ok": True, "status": status, "plan_unchanged": True}

    # ----------------------------
    # Invoices (por ahora no cambian plan)
    # ----------------------------
    if event_type == "invoice.paid":
        return {"ok": True}

    if event_type == "invoice.payment_failed":
        return {"ok": True}

    return {"ok": True}

# =========================
# BILLING (Checkout + Portal)
# =========================

STRIPE_SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL", EVANTIS_APP_URL.rstrip("/") + "/billing/success")
STRIPE_CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", EVANTIS_APP_URL.rstrip("/") + "/billing/cancel")
STRIPE_PORTAL_RETURN_URL = os.getenv("STRIPE_PORTAL_RETURN_URL", EVANTIS_APP_URL.rstrip("/") + "/account")

class CheckoutResponse(BaseModel):
    url: str

class PortalResponse(BaseModel):
    url: str


def db_get_stripe_customer_id(user_id: str) -> Optional[str]:
    with db_conn() as conn:
        row = conn.execute(
            "SELECT stripe_customer_id FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
        return (row[0] or "").strip() if row and row[0] else None


def db_set_stripe_customer_id(user_id: str, customer_id: str) -> None:
    if not customer_id:
        return
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET stripe_customer_id=? WHERE user_id=?",
            (customer_id, user_id),
        )
        conn.commit()


def get_or_create_customer(user_id: str, email: str) -> str:
    # 1) si ya existe en DB, úsalo
    existing = db_get_stripe_customer_id(user_id)
    if existing:
        return existing

    # 2) crea customer con metadata para poder linkear en webhook
    customer = stripe.Customer.create(
        email=email,
        metadata={
            "evantis_user_id": user_id,
            "evantis_env": STRIPE_MODE,
        },
    )
    cid = (customer.get("id") or "").strip()
    db_set_stripe_customer_id(user_id, cid)
    return cid


def price_id_for_plan(plan: str) -> str:
    plan = (plan or "").strip().lower()
    if plan == "pro":
        return STRIPE_PRICE_PRO
    if plan == "premium":
        return STRIPE_PRICE_PREMIUM
    return ""

def _user_email_or_400(user: dict) -> str:
    """
    Obtiene email del objeto 'user' inyectado por require_user().
    Si no viene, intenta recargar desde DB por user_id.
    """
    if not isinstance(user, dict):
        raise HTTPException(status_code=400, detail="Stripe error: usuario inválido (no dict).")

    email = (user.get("email") or "").strip().lower()
    if email:
        return email

    user_id = user.get("user_id") or user.get("id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Stripe error: falta user_id para resolver email.")

    # 🔁 Ajusta el nombre de tu función si se llama distinto:
    # - db_get_user_by_id
    # - db_get_user
    # - db_get_user_safe
    row = None
    if "db_get_user_by_id" in globals():
        row = db_get_user_by_id(user_id)
    elif "db_get_user" in globals():
        row = db_get_user(user_id)

    if not row or not isinstance(row, dict):
        raise HTTPException(status_code=400, detail="Stripe error: no se pudo cargar usuario desde DB.")

    email = (row.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Stripe error: email no disponible para Stripe.")

    return email

class CheckoutRequest(BaseModel):
    plan: Literal["pro", "premium"]

@app.post("/billing/checkout")
def billing_checkout(payload: CheckoutRequest, user: dict = Depends(require_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe no configurado (STRIPE_SECRET_KEY)")

    if not FRONTEND_BASE_URL:
        raise HTTPException(status_code=500, detail="Falta FRONTEND_BASE_URL")

    # (opcional) evitar que free users creen checkout si ya son pro/premium
    # if user.get("plan") in ("pro", "premium"):
    #     raise HTTPException(status_code=400, detail="Ya tienes un plan activo")

    try:
        price_id = stripe_price_for_plan(payload.plan)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Plan inválido: {e}")

    # Success/cancel: vuelve al frontend y fuerza refresh de /auth/me
    success_url = f"{FRONTEND_BASE_URL}/?billing=success&session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{FRONTEND_BASE_URL}/?billing=cancel"

    try:
        # Si ya tenemos customer, lo reutilizamos
        customer_id = user.get("stripe_customer_id")
        if not customer_id:
            # Puedes crear customer explícito para asegurar persistencia
            cust = stripe.Customer.create(
                email=user["email"],
                metadata={"evantis_user_id": user["user_id"], "evantis_env": STRIPE_MODE},
            )
            customer_id = cust["id"]
            db_set_stripe_customer(user["user_id"], customer_id)

        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=True,
            subscription_data={
                "metadata": {
                    "user_id": user["user_id"],
                    "plan": payload.plan,
                }
            },
            metadata={
                "user_id": user["user_id"],
                "plan": payload.plan,
            },
        )
        return {"ok": True, "url": session["url"]}
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=500, detail=f"Stripe error: {getattr(e, 'user_message', str(e))}")

class PortalRequest(BaseModel):
    return_url: Optional[str] = None

@app.post("/billing/portal")
def billing_portal(payload: PortalRequest, user: dict = Depends(require_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe no configurado (STRIPE_SECRET_KEY)")

    if not FRONTEND_BASE_URL:
        raise HTTPException(status_code=500, detail="Falta FRONTEND_BASE_URL")

    customer_id = user.get("stripe_customer_id")

    # Fallback si no está guardado (no ideal, pero útil)
    if not customer_id:
        try:
            lst = stripe.Customer.list(email=user["email"], limit=1)
            if lst and lst.get("data"):
                customer_id = lst["data"][0]["id"]
                db_set_stripe_customer(user["user_id"], customer_id)
        except stripe.error.StripeError:
            customer_id = None

    if not customer_id:
        raise HTTPException(status_code=400, detail="No hay customer Stripe para este usuario (paga o crea checkout primero)")

    return_url = (payload.return_url or FRONTEND_BASE_URL).strip()
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return {"ok": True, "url": session["url"]}
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=500, detail=f"Stripe error: {getattr(e, 'user_message', str(e))}")

# ----------------------------
# Exceptions
# ----------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = None
    try:
        if exc.body is None:
            body = None
        elif isinstance(exc.body, (dict, list, str, int, float, bool)):
            body = exc.body
        elif isinstance(exc.body, (bytes, bytearray)):
            body = exc.body.decode("utf-8", errors="replace")
        else:
            body = str(exc.body)
    except Exception:
        body = "<unserializable>"

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": body,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("🔥 Unhandled exception:", repr(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

@app.on_event("startup")
def on_startup():
    assert_config_or_die()
    db_init()
    assert_stripe_mode()

    # Stripe API key (necesario para Checkout/Portal/Customers/Subscriptions server-side)
    if STRIPE_SECRET_KEY:
        stripe.api_key = STRIPE_SECRET_KEY

    print(f">>> Startup OK | DB={DB_PATH} | STRIPE_MODE={STRIPE_MODE}")

# ----------------------------
# Rate limiting (in-memory, por API key)
# ----------------------------
RATE_LIMIT_CHAT_PER_HOUR = int(os.getenv("RATE_LIMIT_CHAT_PER_HOUR", "100"))
RATE_LIMIT_TEACH_PER_HOUR = int(os.getenv("RATE_LIMIT_TEACH_PER_HOUR", "60"))
_rate_state: dict[str, dict[str, float | int]] = {}


def _rate_limit(bucket: str, limit_per_hour: int, x_api_key: str):
    now = time.time()
    key = f"{bucket}:{x_api_key}"
    st = _rate_state.get(key)
    if not st or now >= float(st["reset"]):
        _rate_state[key] = {"reset": now + 3600, "count": 0}
        st = _rate_state[key]
    st["count"] = int(st["count"]) + 1
    if int(st["count"]) > limit_per_hour:
        seconds_left = int(float(st["reset"]) - now)
        raise HTTPException(
            status_code=429,
            detail=f"Límite excedido ({bucket}): {limit_per_hour}/hora. Intenta de nuevo en {seconds_left}s.",
        )


def rate_limit_chat(x_api_key: str = Depends(require_student_or_admin)):
    _rate_limit("chat", RATE_LIMIT_CHAT_PER_HOUR, x_api_key)


def rate_limit_teach(x_api_key: str = Depends(require_student_or_admin)):
    _rate_limit("teach", RATE_LIMIT_TEACH_PER_HOUR, x_api_key)


# ----------------------------
# NPM rules (tu base)
# ----------------------------
NPM_BASE = [
    "Redactar en español, con alta exigencia académica, precisión terminológica y coherencia interna.",
    "Definir conceptos antes de utilizarlos; evitar saltos lógicos.",
    "Usar Markdown con estructura: H1 (título), Datos (materia/nivel/duración/estilo), índice breve, desarrollo por H2/H3, y cierre con preguntas y errores clínicos frecuentes.",
    "Incluir tablas o listas cuando aumenten claridad.",
    "Mantenerse estrictamente dentro del tema y subtópicos indicados.",
    "Ajustar profundidad al nivel solicitado sin superficialidad.",
]

NPM_PROFILE = {
    "basicas": [
        "Objetivo: estructura, mecanismos y lógica científica; no formar conducta clínica.",
        "Prohibido: diagnóstico, diagnóstico diferencial, tratamiento, algoritmos clínicos o conducta de urgencias.",
        "Enfermedades solo como contexto mecanístico (máximo 2–3 líneas), sin manejo ni guías.",
        "Priorizar definiciones operativas, clasificaciones, rutas, regulación y relaciones causa-efecto.",
        "Usar tablas/cuadros comparativos cuando mejoren claridad (A vs B, tipo I vs II, etc.).",
        "Notación científica correcta cuando aplique (pH, pKa, ΔG, Km, Vmax, unidades).",
        "Incluir errores conceptuales frecuentes y puntos de confusión típicos.",
        "Preguntas de repaso centradas en comprensión mecanística (no casos clínicos).",
    ],
    "puente": [
        "Objetivo: fisiopatología + correlación clínica dirigida sin convertirlo en manejo clínico completo.",
        "Permitir diagnóstico a nivel conceptual: criterios generales, patrones y hallazgos típicos (sin algoritmos exhaustivos).",
        "Diagnóstico diferencial limitado y razonado (3–6 diferenciales clave) solo si aporta a entender el mecanismo.",
        "Tratamiento solo a nivel general (principios/familias terapéuticas), sin dosis ni esquemas; 'primera línea' solo si el usuario lo pide.",
        "Evitar conducta de urgencias y protocolos paso a paso (eso corresponde a clínicas).",
        "Incluir correlación con morfología/laboratorio/imagen cuando aplique (especialmente patología y propedéutica).",
        "Priorizar entidades high-yield manteniendo el eje fisiopatológico.",
        "Cierre con perlas de examen orientadas a fisiopatología (no decisiones terapéuticas).",
    ],
    "clinicas": [
        "Objetivo: razonamiento clínico y toma de decisiones.",
        "Estructura clínica obligatoria completa (11 secciones E-Vantis clínico).",
        "Incluir diagnóstico diferencial, estándar de oro y tamizaje cuando aplique.",
        "Tratamiento acorde al nivel (pregrado/internado), sin dosis exactas salvo solicitud expresa.",
        "Incluir pronóstico, complicaciones y algoritmos diagnósticos/terapéuticos con pasos numerados.",
        "Agregar perlas tipo ENARM/USMLE y errores clínicos frecuentes.",
    ],
}

BASIC_SECTIONS = [
    "Definición y concepto",
    "Clasificación",
    "Estructura / componentes",
    "Mecanismo o función",
    "Regulación / control",
    "Integración con otras materias básicas",
    "Correlación conceptual",
    "Errores conceptuales frecuentes",
    "Preguntas de repaso",
]

BRIDGE_SECTIONS = [
    "Definición",
    "Contexto fisiopatológico",
    "Mecanismo de lesión o alteración",
    "Manifestaciones clínicas",
    "Correlación clínica dirigida",
    "Diagnóstico conceptual",
    "Diagnósticos diferenciales clave",
    "Principios generales de tratamiento",
    "Errores frecuentes de razonamiento",
    "Preguntas de repaso",
]

CLINICAL_SECTIONS = [
    "Definición",
    "Epidemiología y estadística",
    "Cuadro clínico",
    "Signos y síntomas clave",
    "Diagnóstico",
    "Tratamiento",
    "Pronóstico",
    "Complicaciones",
    "Algoritmos de diagnóstico y tratamiento",
    "Preguntas de repaso",
    "Errores clínicos frecuentes",
]


def build_basic_template_instruction() -> str:
    ordered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(BASIC_SECTIONS)])
    return f"""
OBLIGATORIO (E-Vantis básicas): genera el contenido en Markdown usando EXACTAMENTE estos encabezados H2 (##) y en este orden:
{ordered}

Reglas duras:
- No omitas encabezados.
- No reordenes encabezados.
- No cambies el nombre de los encabezados.
- Prohibido incluir diagnóstico clínico, diagnóstico diferencial, tratamiento o algoritmos clínicos.
- La 'Correlación conceptual' debe ser molecular, estructural o fisiológica (NO clínica).
- Usa notación científica correcta cuando aplique.
- Si una sección no se puede desarrollar, ESCRIBE una versión mínima útil (1–3 líneas) sin decir "No aplica".
- PROHIBIDO numerar encabezados H2: no uses '## 1. ...'. Usa EXACTAMENTE los títulos listados.
""".strip()


def build_bridge_template_instruction() -> str:
    ordered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(BRIDGE_SECTIONS)])
    return f"""
OBLIGATORIO (E-Vantis puente): genera el contenido en Markdown usando EXACTAMENTE estos encabezados H2 (##) y en este orden:
{ordered}

Reglas duras:
- No omitas encabezados.
- No reordenes encabezados.
- No cambies el nombre de los encabezados.
- Diagnóstico SOLO conceptual (no operativo).
- Diagnósticos diferenciales limitados (3–6).
- Tratamiento SOLO en principios generales (sin dosis ni esquemas).
- Prohibido incluir algoritmos clínicos.
- Si una sección no se puede desarrollar, ESCRIBE una versión mínima útil (1–3 líneas) sin decir "No aplica".
- PROHIBIDO numerar encabezados H2: no uses '## 1. ...'. Usa EXACTAMENTE los títulos listados.
""".strip()


def build_clinical_template_instruction() -> str:
    ordered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(CLINICAL_SECTIONS)])
    return f"""
OBLIGATORIO (E-Vantis clínico): genera el contenido en Markdown usando EXACTAMENTE estos encabezados H2 (##) y en este orden:
{ordered}

Reglas duras:
- No omitas encabezados.
- No reordenes encabezados.
- No cambies el nombre de los encabezados.
- NO se permite escribir "No aplica" en ninguna sección. Si algo no aplica al tema, ADAPTA con 1–3 líneas útiles.
- En 'Diagnóstico' incluye explícitamente: Enfoque diagnóstico, Diagnósticos diferenciales, Estándar de oro, Tamizaje.
- En 'Algoritmos de diagnóstico y tratamiento' incluye:
  - Algoritmo diagnóstico (pasos 1..n)
  - Algoritmo terapéutico (pasos 1..n)
- PROHIBIDO numerar encabezados H2.
""".strip()


# ----------------------------
# FASE 4 — Convención editorial (una sola vez, global)
# ----------------------------
PHASE4_MD_CONVENTION_V1 = """
# CONVENCIÓN MARKDOWN CLÍNICA — E-VANTIS v1 (FASE 4 EXPERIMENTAL)

Si editorial_v1=true, DEBES usar estas marcas correctamente (sin relleno):

A) HIGH-YIELD (no cruza líneas; máximo 5–8 por documento):
==texto==

B) BADGES (solo en headings H2; máximo 2 por sección; SIEMPRE al inicio):
## [badge:alta_prioridad] Diagnóstico
Badges soportados: alta_prioridad, concepto_clave, red_flag, error_frecuente, enfoque_enarm

C) CALLOUTS (blockquote explícito; primera línea EXACTA):
> [callout:perla_clinica]
> Texto del callout...
Callouts soportados: perla_clinica, advertencia, punto_de_examen, razonamiento_clinico

Reglas duras:
- No inventes badges/callouts fuera del set.
- Mantén Markdown válido.
- Mantén la estructura E-Vantis sin cambiar nombres ni orden de secciones.
""".strip()


# ----------------------------
# Helpers: headings (H2) + validación clínica
# ----------------------------
def _extract_h2_headings(md: str) -> List[str]:
    raw = [h.strip() for h in re.findall(r"^##\s+(.+?)\s*$", md, flags=re.MULTILINE)]
    cleaned: List[str] = []
    for h in raw:
        # Elimina uno o más badges al inicio: [badge:...]
        # (PATCH: tolera espacios: [ badge : red_flag ])
        h2 = re.sub(
            r"^\s*(?:\[\s*badge\s*:\s*[a-z0-9_-]+\s*\]\s*)+",
            "",
            h,
            flags=re.IGNORECASE,
        ).strip()
        cleaned.append(h2)
    return cleaned


def build_system_npm(profile: str) -> list[str]:
    base = list(NPM_BASE)
    base += list(NPM_PROFILE.get(profile, []))
    return base


def _norm_txt(s: str) -> str:
    # PATCH: normaliza acentos para evitar falsos negativos en validación
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


def validate_clinical_markdown(md: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    headings = _extract_h2_headings(md)
    if [h.strip() for h in headings] != CLINICAL_SECTIONS:
        return False, [
            "Encabezados H2 no coinciden EXACTAMENTE con el estándar clínico E-Vantis.",
            f"Esperado: {CLINICAL_SECTIONS}",
            f"Encontrado: {headings}",
        ]

    if re.search(r"(?i)\bno aplica\b", md):
        errors.append("Prohibido usar 'No aplica' en clases clínicas.")

    mdn = _norm_txt(md)
    must_terms = ["enfoque diagnostico", "diagnosticos diferenciales", "estandar de oro", "tamizaje"]
    missing = [t for t in must_terms if t not in mdn]
    if missing:
        errors.append("En 'Diagnóstico' faltan explícitamente: " + ", ".join(missing))

    if "algoritmos de diagnóstico y tratamiento" not in md.lower():
        errors.append("Falta la sección 'Algoritmos de diagnóstico y tratamiento'.")
    else:
        parts = re.split(r"(?m)^##\s+", md)
        algo_body = ""
        for p in parts:
            if p.lower().startswith("algoritmos de diagnóstico y tratamiento"):
                algo_body = p
                break

        if not algo_body:
            errors.append("No se pudo aislar el contenido de 'Algoritmos de diagnóstico y tratamiento'.")
        else:
            algo_lower = algo_body.lower()

            if "algoritmo diagnóstico" not in algo_lower:
                errors.append("En algoritmos falta 'Algoritmo diagnóstico'.")
            if ("algoritmo terapéutico" not in algo_lower) and ("algoritmo de tratamiento" not in algo_lower):
                errors.append("En algoritmos falta 'Algoritmo terapéutico' o 'Algoritmo de tratamiento'.")

            if not re.search(r"(?m)^\s*1\.\s+", algo_body):
                errors.append("En algoritmos faltan pasos numerados (1., 2., 3., ...).")

    return (len(errors) == 0), errors


def build_evantis_header_instruction(subject_name: str, level: str, duration: int, style: str) -> str:
    return f"""
FORMATO OBLIGATORIO DE INICIO (NO OMITIR):

1. Inicia SIEMPRE con un título H1 (#) con el nombre de la clase.
2. Inmediatamente después incluye EXACTAMENTE este bloque:

**Materia:** {subject_name}  
**Nivel:** {level}  
**Duración:** {duration} minutos  
**Estilo:** {style}

3. Luego incluye un encabezado H2: ## Índice
4. El índice debe listar TODAS las secciones obligatorias en orden.
""".strip()


# ----------------------------
# GPC summary
# ----------------------------
def build_gpc_summary_prompt(subject_name: str, topic_name: str) -> str:
    return f"""
Genera un RESUMEN basado en Guías de Práctica Clínica (GPC) mexicanas vigentes.
El contenido debe ser ORIGINAL, EDUCATIVO y NO copiar texto literal de ninguna guía.

Materia: {subject_name}
Tema ENARM: {topic_name}

REGLA CRÍTICA:
- DEBES usar web_search para identificar una GPC mexicana pertinente y vigente.
- NO inventes enlaces. Si no encuentras URL exacta, escribe "Enlace: no disponible en la consulta."
- NO omitas la sección "## Validación de la GPC consultada".

Estructura OBLIGATORIA (Markdown), en este orden:

## Puntos clave ENARM
## Diagnóstico y criterios
## Conducta / primera línea (principios)
## Red flags / criterios de referencia
## Algoritmo práctico (pasos numerados)
## Validación de la GPC consultada

Dentro de "## Validación de la GPC consultada" incluye OBLIGATORIAMENTE estas líneas EXACTAS (con dos puntos):
- Nombre: <nombre oficial exacto de la GPC>
- Año: <YYYY>
- Institución: <CENETEC/SSA/IMSS/ISSSTE/etc.>
- Última actualización: <YYYY-MM-DD o Mes YYYY o "no especificada en la fuente consultada">
- Enlace: <URL exacta> o "no disponible en la consulta."

## Justificación de pertinencia (OBLIGATORIA)
En 1–2 líneas, explica explícitamente por qué la GPC consultada corresponde directamente
al tema solicitado (**{topic_name}**) y no a otro problema clínico distinto.

Reglas duras:
- Prohibido copiar o parafrasear texto literal de GPC.
- Todo debe estar redactado con lenguaje académico propio.
""".strip()


def validate_gpc_summary(md: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    required_section = "## Validación de la GPC consultada"

    if required_section not in md:
        errors.append(f"Falta sección obligatoria: {required_section}.")
        return False, errors

    tail = md.split(required_section, 1)[-1].lower()

    required_terms = ["nombre", "año", "institución", "última actualización"]
    missing = [t for t in required_terms if t not in tail]
    if missing:
        errors.append("En validación faltan datos: " + ", ".join(missing))

    if "última actualización" not in tail:
        errors.append("Debe incluir explícitamente 'Última actualización: ...'.")

    if "justificación" not in md.lower():
        errors.append("Falta la justificación de pertinencia de la GPC.")

    if "http" in md.lower() and "validación de la gpc consultada" not in md.lower():
        errors.append("Incluye enlaces sin sección de validación de GPC.")

    return (len(errors) == 0), errors


def _resp_used_web_search(r) -> bool:
    try:
        out = getattr(r, "output", None) or []
        for item in out:
            t = getattr(item, "type", "") or ""
            if t in ("web_search_call", "web_search"):
                return True

            content = getattr(item, "content", None) or []
            for c in content:
                ct = getattr(c, "type", "") or ""
                if ct in ("web_search_call", "web_search"):
                    return True
    except Exception:
        pass

    return False


def build_repair_instruction(errors: List[str]) -> str:
    bullets = "\n".join([f"- {e}" for e in errors])
    ordered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(CLINICAL_SECTIONS)])
    return f"""
Tu respuesta anterior NO cumple el estándar clínico E-VANTIS por estas razones:
{bullets}

REPARA y devuelve SOLO el Markdown final, cumpliendo:
- Encabezados H2 (##) EXACTOS y en el orden:
{ordered}
- PROHIBIDO escribir "No aplica".
- Diagnóstico debe incluir explícitamente: Enfoque diagnóstico, Diagnósticos diferenciales, Estándar de oro, Tamizaje.
- Algoritmos debe incluir: Algoritmo diagnóstico + Algoritmo terapéutico con pasos 1..n
""".strip()


# ----------------------------
# Subject rules (IMPORTANTE)
# ----------------------------
NPM_SUBJECT_RULES = {
    "inmunologia": [
        "Se permite mencionar enfermedades y contextos clínicos SOLO para justificar indicaciones.",
        "Incluir terapias inmunológicas modernas (p. ej., mAbs, checkpoint inhibitors, CAR-T, vacunas, etc.) explicando: diana terapéutica, mecanismo, indicaciones clínicas típicas y riesgos generales.",
        "No dar esquemas de manejo (dosis, algoritmos terapéuticos, líneas de tratamiento).",
    ],
    "microbiologia": [
        "Microbiología es una materia clínica integral: TODA clase debe incluir cuadro clínico completo, diagnóstico, tratamiento farmacológico específico (si existe) y prevención/vacunas.",
        "Incluir siempre: nombre de la enfermedad, sintomatología, signos, periodo de incubación, tiempo de evolución y complicaciones.",
        "Incluir factores de riesgo y factores protectores de forma explícita.",
        "Tratamiento farmacológico: permitir fármacos de elección y alternativas con su mecanismo de acción; NO incluir dosis exactas ni esquemas detallados salvo solicitud expresa.",
        "Diferenciar claramente colonización vs infección y portador vs enfermedad activa cuando aplique.",
        "Prevención es obligatoria: vacunas, profilaxis y medidas de control.",
        "Integrar perlas ENARM/USMLE y preguntas tipo examen con razonamiento clínico.",
        "Evitar farmacología profunda (farmacocinética, ajustes finos); mantener enfoque microbiológico-clínico.",
    ],
    "farmacologia": [
        "Incluir definiciones base y clasificaciones: por grupo farmacológico, por mecanismo y por indicación clínica (p. ej. hipoglucemiantes, antieméticos, antipiréticos).",
        "Para cada fármaco: mecanismo de acción, indicaciones, contraindicaciones, efectos adversos y secundarios, embarazo y lactancia, interacciones relevantes.",
        "Incluir farmacocinética esencial: vida media y consideraciones de eliminación; incluir dosis tóxica/letal SOLO como concepto (números solo si están estandarizados o si el usuario lo pide).",
    ],
    "fisiologia": [
        "Profundidad alta y énfasis en integración con histología, bioquímica, anatomía y embriología.",
        "Priorizar mecanismos, retroalimentación, curvas/relaciones y predicción fisiológica.",
    ],
    "intro_cirugia": [
        "Enfoque en fundamentos quirúrgicos: asepsia/antisepsia, instrumentación, suturas, cicatrización, seguridad del paciente, principios perioperatorios.",
        "Patologías SOLO si están en el temario; sin convertirlo en materia clínica de manejo extenso.",
    ],
    "anatomia_patologica_1": [
        "Definir siempre la lesión antes de describir variantes.",
        "Explicar correlación morfológica con manifestaciones clínicas.",
        "Incluir mecanismos fisiopatológicos cuando correspondan.",
        "Usar terminología histopatológica estándar.",
        "No mezclar temas de otras materias.",
        "Priorizar entidades de relevancia clínica.",
        "Mantener enfoque diagnóstico y pronóstico.",
    ],
    "anatomia_patologica_2": [
        "Definir siempre la lesión antes de describir variantes.",
        "Explicar correlación morfológica con manifestaciones clínicas.",
        "Incluir mecanismos fisiopatológicos cuando correspondan.",
        "Usar terminología histopatológica estándar.",
        "No mezclar temas de otras materias.",
        "Priorizar entidades de relevancia clínica.",
        "Mantener enfoque diagnóstico y pronóstico.",
    ],
    "cardiologia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones).",
        "Diagnóstico obligatorio con: enfoque diagnóstico, diagnósticos diferenciales, estándar de oro, tamizaje (si aplica).",
        "Tratamiento obligatorio acorde al nivel (pregrado/internado).",
        "Pronóstico y complicaciones obligatorios.",
        "Cierre obligatorio con algoritmos de diagnóstico y tratamiento (pasos numerados).",
        "Enfoque en razonamiento clínico y toma de decisiones.",
        "Lenguaje médico formal; evitar relleno.",
        "No se permiten clases sin estructura clínica completa.",
    ],
    "dermatologia": [
        "Materia clínica: aplicar SIEMPRE la plantilla clínica oficial E-Vantis con los 11 encabezados obligatorios y en el orden exacto.",
        "En 'Diagnóstico' incluir explícitamente: Enfoque diagnóstico, Diagnósticos diferenciales, Estándar de oro, Tamizaje (aplica/no aplica).",
        "En 'Algoritmos' incluir dos algoritmos en pasos numerados: (1) diagnóstico, (2) tratamiento.",
        "Siempre iniciar el razonamiento dermatológico por: morfología primaria/ secundaria, distribución, topografía, patrón y evolución temporal.",
        "Diferenciar consistentemente: infeccioso vs inflamatorio vs autoinmune vs neoplásico vs farmacológico.",
        "Incluir 'red flags' dermatológicas cuando aplique (p. ej., fiebre, dolor desproporcionado, compromiso mucoso, necrosis, inmunosupresión).",
        "Indicar cuándo usar: examen directo KOH, cultivo micológico/bacteriano, Tzanck (si aplica), dermatoscopia, biopsia y anatomía patológica.",
        "En lesiones pigmentadas: describir criterios ABCDE y cuándo derivar/biopsiar.",
        "En infecciones cutáneas: diferenciar colonización vs infección; impétigo/foliculitis/celulitis/erisipela con criterios clínicos claros.",
        "Tratamiento acorde a nivel pregrado: incluir primera línea y alternativas por escenarios frecuentes (alergias, embarazo si relevante, pediatría si relevante), SIN dosis numéricas salvo que el usuario las solicite.",
        "Incluir medidas generales obligatorias cuando aplique: higiene, barrera cutánea, emolientes, fotoprotección, evitar desencadenantes, educación del paciente.",
        "Separar tratamiento tópico vs sistémico y dar criterios clínicos para escalar.",
        "Incluir medidas preventivas cuando el tema lo requiera (vacunas, control de contagio, medidas de contacto, profilaxis si aplica).",
        "Complicaciones deben ser concretas y clínicas (p. ej., sepsis, cicatriz, postinflamatoria, nefritis postestreptocócica, compromiso ocular, etc.).",
        "Pronóstico debe incluir: curso esperado, recurrencia, y factores de mal pronóstico.",
        "Agregar perlas high-yield ENARM/USMLE solo si son realmente discriminativas del diagnóstico o manejo (no relleno).",
        "Preguntas de repaso: 8–12 con enfoque en reconocimiento de lesiones, diferenciales clave y decisión terapéutica.",
        "Errores clínicos frecuentes: 5–8 orientados a fallas reales (confundir celulitis vs dermatitis, omitir mucosas en SJS/TEN, etc.).",
        "No dar recomendaciones para auto-tratamiento del público general; el enfoque es médico-académico.",
        "Cuando exista potencial gravedad (SJS/TEN, fascitis necrotizante, meningococcemia, eritrodermia, anafilaxia por urticaria/angioedema, etc.), incluir conducta de urgencia y criterios de referencia.",
    ],
    "endocrinologia": [
        "Generar clases clínicas con razonamiento endocrinológico: clínica + fisiopatología + interpretación de pruebas + conducta.",
        "Siempre diferenciar: TAMIZAJE vs CONFIRMACIÓN vs CLASIFICACIÓN vs LOCALIZACIÓN (cuando aplique).",
        "Siempre interpretar con contexto clínico y probabilidad pretest; evitar decisiones basadas en un solo dato.",
        "Incluir errores/artefactos frecuentes: Interferencias analíticas (p. ej., biotina en pruebas tiroideas; hemólisis, lipemia). Variabilidad biológica y ritmo circadiano (p. ej., cortisol; prolactina; testosterona).",
        "Señalar RED FLAGS y criterios de urgencia: Hipoglucemia grave, CAD (cetoacidosis diabética), EHH (estado hiperosmolar).Crisis suprarrenal, tormenta tiroidea, mixedema. Feocromocitoma con crisis adrenérgica.",
        "Tratamiento: usar enfoque escalonado y seguro. Incluir contraindicaciones y precauciones clave.",
        "No recomendar automanejo; indicar evaluación médica cuando aplique y remarcar urgencias cuando existan banderas rojas.",
    ],
    "farmacologia_clinica": [
        "Materia clínica (farmacoterapia). Estructura clínica completa obligatoria (11 secciones).",
        "NO escribir 'No aplica' en ninguna sección. Si el tema no es una enfermedad, convertir la sección a un equivalente farmacoterapéutico válido.",
        "Epidemiología y estadística: enfocar en magnitud del problema (seguridad del paciente, eventos adversos por medicación) sin números no citados; preferir enunciados cualitativos.",
        "Cuadro clínico: describir presentaciones típicas del error de prescripción y problemas relacionados con medicamentos (PRM): falta de eficacia, toxicidad, interacción, alergia, duplicidad, omisión.",
        "Signos y síntomas clave: incluir red flags farmacoterapéuticas (anafilaxia, sangrado, depresión respiratoria, hepatotoxicidad, rabdomiólisis, QT prolongado, hipoglucemia, etc.) y datos del paciente relevantes (IR/IH, embarazo, alergias, polifarmacia).",
        "Diagnóstico: obligatorio incluir enfoque diagnóstico + diferenciales + estándar de oro + tamizaje (si aplica) Y además criterios explícitos para iniciar, ajustar, cambiar o suspender tratamiento.",
        "Tratamiento: obligatorio incluir reconciliación de medicamentos, selección de terapia basada en evidencia, seguridad, interacciones, plan de monitoreo y educación al paciente.",
        "Algoritmos: deben incluir pasos numerados e incluir explícitamente reconciliación, verificación de alergias/interacciones, ajuste por función renal/hepática, y plan de seguimiento.",
        "Prohibido usar porcentajes o cifras epidemiológicas si no se citan fuentes; preferir lenguaje cualitativo.",
        "En la sección Tamizaje, describir tamizajes/controles pre-tratamiento (renal, hepático, embarazo, QT, etc.), sin decir ‘no aplica’.",
    ],
    "gastroenterologia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones): Definición; Epidemiología y estadística; Cuadro clínico; Signos y síntomas clave; Diagnóstico; Tratamiento; Pronóstico; Complicaciones; Algoritmos de diagnóstico y tratamiento; Preguntas de repaso; Errores clínicos frecuentes.",
        "La sección de Diagnóstico es obligatoria e incluye explícitamente: enfoque diagnóstico, diagnósticos diferenciales, estándar de oro y tamizaje (si aplica).",
        "Enfoque de gastroenterología con razonamiento clínico y toma de decisiones: priorizar identificación de gravedad, triage, urgencias digestivas y criterios de referencia.",
        "Obligatorio diferenciar: patología funcional vs orgánica; inflamatoria vs infecciosa vs neoplásica; obstructiva vs no obstructiva; alta vs baja (cuando aplique).",
        "Obligatorio integrar red flags / datos de alarma digestivos: pérdida de peso, anemia, sangrado, disfagia progresiva, vómito persistente, ictericia, fiebre, dolor abdominal severo, signos peritoneales, deshidratación, hipotensión/síncope.",
        "Tratamiento obligatorio acorde a nivel pregrado: no dar esquemas avanzados de subespecialidad; sí incluir medidas generales, manejo inicial, terapias de primera línea, contraindicaciones y criterios de escalamiento/referencia.",
        "En temas de Hemorragia gastrointestinal, Síndrome abdominal agudo, pancreatitis, colangitis, hepatitis fulminante o sepsis: incluir abordaje ABC, reanimación inicial, criterios de choque, y manejo inicial seguro. Priorizar seguridad del paciente.",
        "En patologías infecciosas gastrointestinales: incluir enfoque sindromático (diarrea aguda vs crónica; inflamatoria vs no inflamatoria), hidratación, criterios de antibiótico, y prevención (higiene, vacunas cuando aplique). Evitar antibióticos innecesarios.",
        "En hepatología: integrar patrón de daño hepático (hepatocelular vs colestásico vs mixto), evaluación de insuficiencia hepática, complicaciones y criterios de gravedad. Incluir MASLD/ALD/DILI como diferenciales cuando corresponda.",
        "En páncreas y vías biliares: incluir diagnóstico diferencial de dolor epigástrico/hipocondrio derecho, interpretación clínica de enzimas hepáticas/pancreáticas y uso racional de imagen (US, TC, MRCP/CPRE según indicación).",
        "Uso racional de estudios: laboratorio (BH, PFH, PFHepática, electrolitos, amilasa/lipasa, marcadores inflamatorios cuando aplique), heces (copro, coprocultivo cuando aplique), endoscopia (indicación y estándar de oro cuando corresponda) e imagen (US/TC).",
        "En cánceres GI (esófago, estómago, colon, hígado, páncreas): incluir factores de riesgo, signos de alarma, estándar de oro diagnóstica (endoscopia/biopsia cuando aplique), y tamizaje (por ejemplo, cáncer colorrectal) con enfoque práctico.",
        "En ERGE/dispepsia: incluir estrategia de manejo inicial, criterios de endoscopia, y enfoque de Helicobacter pylori cuando aplique (pruebas diagnósticas y control post-tratamiento según práctica clínica).",
        "No se permite lenguaje tipo 'no aplica' en secciones obligatorias: adaptar el contenido al tema (por ejemplo, describir epidemiología de errores/PRM no aplica aquí; en GI siempre hay epidemiología clínica relevante).",
        "Lenguaje médico formal, preciso y sin relleno. Evitar afirmaciones numéricas no sustentadas; si no se citará una fuente, preferir lenguaje cualitativo o rangos generales ampliamente aceptados con fuente.",
        "Cierre obligatorio con 'Algoritmos de diagnóstico y tratamiento' con pasos numerados; debe incluir explícitamente un algoritmo diagnóstico y un algoritmo terapéutico.",
        "Incluir al final: Preguntas de repaso (5–10) y Errores clínicos frecuentes (5–10), orientados a seguridad del paciente y sesgos cognitivos comunes en GI.",
        "Evitar 'no aplica'. Si un apartado es transversal, adaptar con enfoque clínico (frecuencia, grupos de riesgo, carga clínica), sin inventar cifras.",
    ],
    "hematologia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones): Definición; Epidemiología y estadística; Cuadro clínico; Signos y síntomas clave; Diagnóstico; Tratamiento; Pronóstico; Complicaciones; Algoritmos de diagnóstico y tratamiento; Preguntas de repaso; Errores clínicos frecuentes.",
        "Diagnóstico obligatorio con: enfoque diagnóstico, diagnósticos diferenciales, estándar de oro y tamizaje (si aplica). En hematología, incluir siempre: interpretación de biometría hemática, frotis periférico y estudios de médula ósea cuando corresponda.",
        "Enfoque hematológico con razonamiento clínico y toma de decisiones: integrar fisiopatología, correlación clínica y priorización por gravedad (triage) en urgencias hematológicas.",
        "Obligatorio diferenciar y explicitar: anemia microcítica vs normocítica vs macrocítica; hemólisis vs no hemólisis; central (médula) vs periférico (destrucción/consumo); trombocitopenia por producción vs consumo vs secuestro; leucocitosis reactiva vs neoplásica; sangrado primario (plaquetas) vs secundario (coagulación).",
        "Obligatorio integrar datos de alarma hematológicos (red flags): sangrado activo o incontrolable, inestabilidad hemodinámica, signos de choque, hemoptisis/hematemesis/melena/hematuria con compromiso, púrpura extensa, alteración neurológica, fiebre en neutropenia, anemia sintomática severa, dolor óseo intenso con síntomas B, esplenomegalia marcada, sospecha de leucostasis, sospecha de CID, sospecha de TTP/HUS.",
        "Uso racional de estudios: BH con índices eritrocitarios, reticulocitos, ferritina/Fe/TIBC, B12/folato, LDH, bilirrubina indirecta, haptoglobina, Coombs directo, TP/INR, TTPa, fibrinógeno, dímero D, pruebas de función hepática/renal, EGO cuando aplique. Incluir frotis periférico como herramienta clave. Solicitar médula ósea (aspirado/biopsia) cuando el problema sugiera falla medular, citopenias no explicadas o sospecha de neoplasia hematológica.",
        "Estándares de oro (cuando aplique) deben declararse sin ambigüedad: frotis periférico orienta; aspirado/biopsia de médula ósea confirma muchas entidades medulares; citometría de flujo para fenotipo en leucemias/linfomas; citogenética/molecular según sospecha; biopsia ganglionar excisional para linfoma; electroforesis e inmunofijación + cadenas ligeras para mieloma (según enfoque docente).",
        "Tratamiento obligatorio acorde a nivel pregrado: incluir manejo inicial, medidas generales, primera línea, contraindicaciones, monitoreo, y criterios de escalamiento/referencia. Evitar esquemas avanzados de subespecialidad (p. ej., protocolos quimioterapéuticos detallados), pero sí indicar cuándo referir y qué estabilizar primero.",
        "Transfusión e inmuno-hematología: cuando el tema lo requiera, incluir umbrales y decisiones de transfusión SOLO si se citan fuentes en la misma línea; si no se citará fuente, describir criterios clínicos cualitativos (anemia sintomática, sangrado activo, etc.). Incluir compatibilidad, pruebas cruzadas, Coombs, fenotipado cuando aplique, y vigilancia de reacciones transfusionales.",
        "Urgencias hematológicas (obligatorio cuando corresponda: neutropenia febril, sangrado mayor, CID, TTP/HUS, crisis aplásica, leucostasis, hiperviscosidad, síndrome compartimental por hemorragia): incluir enfoque ABC, reanimación, criterios de choque, aislamiento si neutropenia, antibiótico empírico temprano en neutropenia febril, y criterios de referencia inmediata.",
        "Neoplasias hematológicas (SMD, MPN, leucemias agudas/crónicas, linfomas, mieloma): incluir clasificación OMS como marco, características clínicas, métodos diagnósticos (citometría/citogenética/molecular/biopsia según entidad), factores pronósticos a nivel conceptual (sin scores numéricos si no hay fuente), y tratamiento inicial/medidas generales + referencia.",
        "No se permite 'no aplica' en secciones obligatorias: adaptar el contenido a hematología (por ejemplo, en Epidemiología usar lenguaje cualitativo si no se citarán cifras; en Diagnóstico siempre hay estándar de oro y diferenciales).",
        "Lenguaje médico formal, preciso y sin relleno. Evitar afirmaciones numéricas no sustentadas; si no se citará una fuente, preferir lenguaje cualitativo o rangos generales ampliamente aceptados con fuente.",
        "Cierre obligatorio con 'Algoritmos de diagnóstico y tratamiento' con pasos numerados; debe incluir explícitamente un algoritmo diagnóstico y un algoritmo terapéutico.",
        "Incluir al final: Preguntas de repaso (5–10) y Errores clínicos frecuentes (5–10), orientados a seguridad del paciente, sesgos cognitivos (anclaje, cierre prematuro) y errores típicos (no pedir reticulocitos, no revisar frotis, omitir hemólisis, no descartar sangrado oculto, no reconocer neutropenia febril)."
    ],
    "nefrologia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones): Definición; Epidemiología y estadística; Cuadro clínico; Signos y síntomas clave; Diagnóstico; Tratamiento; Pronóstico; Complicaciones; Algoritmos de diagnóstico y tratamiento; Preguntas de repaso; Errores clínicos frecuentes.",
        "Diagnóstico obligatorio con enfoque sindromático y fisiopatológico: prerrenal, intrínseco y postrenal cuando aplique.",
        "Integrar siempre interpretación clínica de laboratorio renal: creatinina, TFG, EGO, electrolitos, gasometría cuando corresponda.",
        "Diferenciar patología aguda vs crónica y establecer criterios de gravedad y urgencia.",
        "En lesión renal aguda, trastornos hidroelectrolíticos y ácido-base: incluir abordaje ABC, estabilización inicial y criterios de hospitalización.",
        "Tratamiento acorde a nivel pregrado: medidas generales, manejo inicial, corrección de causas reversibles y criterios claros de referencia.",
        "Incluir criterios KDIGO cuando apliquen (LRA, ERC).",
        "Evitar esquemas avanzados de subespecialidad; priorizar seguridad del paciente.",
        "Lenguaje médico formal, preciso y sin relleno.",
        "Cierre obligatorio con algoritmos diagnósticos y terapéuticos numerados.",
        "Incluir Preguntas de repaso (5–10) y Errores clínicos frecuentes (5–10)."
    ],
    "neumologia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones): Definición; Epidemiología y estadística; Cuadro clínico; Signos y síntomas clave; Diagnóstico; Tratamiento; Pronóstico; Complicaciones; Algoritmos de diagnóstico y tratamiento; Preguntas de repaso; Errores clínicos frecuentes.",
        "El diagnóstico debe incluir: enfoque clínico, diagnósticos diferenciales, estándar de oro y tamizaje cuando aplique.",
        "Enfoque de neumología clínica con razonamiento fisiopatológico y toma de decisiones.",
        "Obligatorio identificar criterios de gravedad, urgencia y referencia.",
        "Integrar siempre gasometría, espirometría, imagen y laboratorio cuando aplique.",
        "Tratamiento acorde a nivel pregrado: manejo inicial, estabilización, primera línea y criterios de escalamiento.",
        "En patologías agudas: incluir abordaje ABC y manejo inicial seguro.",
        "Uso racional de estudios diagnósticos.",
        "Lenguaje médico formal, preciso y sin relleno.",
        "Cierre obligatorio con algoritmos diagnóstico y terapéutico numerados.",
        "Incluir preguntas de repaso y errores clínicos frecuentes enfocados en seguridad del paciente."
    ],
    "neurologia": [
        "Generar clases clínicas con razonamiento neurológico: clínica + neuroanatomía funcional + fisiopatología + correlación topográfica + conducta.",
        "Siempre orientar el razonamiento respondiendo explícitamente: ¿dónde está la lesión?, ¿cuál es el mecanismo?, ¿qué síndrome neurológico explica el cuadro?",
        "Siempre diferenciar y señalar cuando aplique: lesión CENTRAL vs PERIFÉRICA; neurona motora SUPERIOR vs INFERIOR; compromiso cortical, subcortical, troncoencefálico o medular.",
        "Integrar semiología neurológica relevante al tema: estado de conciencia, pares craneales, fuerza, tono, reflejos, sensibilidad, coordinación, marcha, lenguaje y funciones superiores.",
        "Interpretar estudios complementarios con contexto clínico; justificar indicación de TC vs RM, punción lumbar, EEG y estudios electrofisiológicos; evitar solicitudes indiscriminadas.",
        "Incluir errores y trampas diagnósticas frecuentes: síntomas neurológicos funcionales, variabilidad del examen neurológico, falsos déficits por fatiga, fármacos o alteraciones metabólicas.",
        "Señalar RED FLAGS y criterios de urgencia neurológica: déficit focal agudo, alteración del estado de conciencia, cefalea súbita intensa, convulsión de novo, signos de hipertensión endocraneana, fiebre con rigidez de nuca.",
        "Tratamiento: usar enfoque escalonado, seguro y basado en objetivos; diferenciar manejo agudo, crónico y prevención secundaria; evitar dosis específicas salvo solicitud explícita.",
        "No recomendar automanejo; indicar valoración médica y referencia urgente cuando existan banderas rojas o compromiso neurológico potencialmente reversible."
    ],
    "oftalmologia": [
        "Generar clases clínicas con razonamiento oftalmológico: clínica + anatomía funcional del ojo + fisiopatología + correlación con exploración ocular + conducta.",
        "Siempre estructurar el razonamiento respondiendo explícitamente: ¿cuál es la localización (segmento anterior, posterior, nervio óptico, órbita)?, ¿cuál es el mecanismo (inflamatorio, infeccioso, vascular, traumático, degenerativo)?, ¿cuál es el diagnóstico sindromático principal?",
        "Siempre integrar exploración dirigida y su interpretación: agudeza visual, reflejos pupilares, motilidad ocular, biomicroscopía/lámpara de hendidura cuando aplique, fondo de ojo, PIO, y pruebas básicas (fluoresceína, oftalmoscopía).",
        "Siempre diferenciar URGENCIA vs NO URGENCIA: pérdida visual súbita, dolor ocular intenso, ojo rojo doloroso, fotofobia, trauma ocular, leucocoria, defecto pupilar aferente, y síntomas neurológicos acompañantes como RED FLAGS.",
        "Interpretar estudios complementarios con contexto clínico y uso racional: tonometría, campimetría, OCT, angiografía, ultrasonido ocular, y neuroimagen cuando se sospeche patología de órbita o neuro-oftalmológica.",
        "Incluir errores y trampas frecuentes: confundir conjuntivitis con queratitis/uveítis/glaucoma agudo; subestimar dolor ocular con fotofobia; omitir agudeza visual y pupilas; no evertir párpado en cuerpo extraño; retrasar referencia en pérdida visual súbita.",
        "Señalar criterios de referencia urgente: sospecha de glaucoma agudo, desprendimiento de retina, oclusiones vasculares retinianas, neuritis óptica/papiledema, endoftalmitis, queratitis grave (incluida por lentes de contacto), trauma penetrante o químico.",
        "Tratamiento: usar enfoque escalonado y seguro a nivel pregrado; describir medidas iniciales y de soporte, contraindicaciones clave y cuándo NO iniciar esteroides o anestésicos tópicos sin valoración especializada.",
        "Evitar recomendaciones de automanejo; indicar evaluación médica/oftalmológica y urgencias cuando existan banderas rojas o riesgo de pérdida visual."
    ],
    "otorrinolaringologia": [
        "Generar clases clínicas con razonamiento otorrinolaringológico: clínica + anatomía funcional + fisiopatología + exploración ORL dirigida + interpretación de pruebas + conducta.",
        "Siempre localizar el problema por región (cavidad oral/faringe, laringe, nariz/senos paranasales, oído, cuello) y definir mecanismo probable (infeccioso, inflamatorio, alérgico, traumático, obstructivo, tumoral, vascular).",
        "Integrar exploración física sistemática y su interpretación: inspección orofaríngea, palpación cervical, otoscopía, rinoscopía/anterior, evaluación de voz y deglución, pruebas básicas de audición (susurro/diapasón: Weber-Rinne), y evaluación neurológica focal cuando aplique (pares craneales).",
        "Siempre diferenciar: manejo ambulatorio vs URGENCIA/REFERENCIA. Señalar RED FLAGS: compromiso de vía aérea (estridor, disnea, tiraje, sialorrea, incapacidad para deglutir), hemorragia activa significativa (epistaxis no controlable), sepsis/estado tóxico, odinofagia severa con trismus, absceso periamigdalino/retrofaríngeo, otitis complicada (mastoiditis), vértigo con signos neurológicos, parálisis facial periférica de inicio agudo, masa cervical de alto riesgo.",
        "Diagnóstico: usar enfoque sindromático y etiológico. En infecciones, diferenciar viral vs bacteriana y complicaciones; en rinitis/rinosinusitis, diferenciar alérgica vs infecciosa; en oído, diferenciar conductiva vs neurosensorial; en cuello, diferenciar congénita vs inflamatoria/infecciosa vs tumoral.",
        "Uso racional de estudios: BH y marcadores solo si cambian conducta; cultivo/pruebas rápidas cuando aplique; nasofibrolaringoscopía y laringoscopía para disfonía/disfagia persistente o sospecha tumoral; imagen (US, TC, RM) solo con indicación clínica (complicaciones, abscesos, masas, trauma).",
        "Incluir errores y trampas frecuentes: tratar indiscriminadamente con antibióticos sin criterios; omitir evaluación de vía aérea en odinofagia severa; subestimar epistaxis posterior; no diferenciar hipoacusia conductiva vs neurosensorial; no reconocer mastoiditis o complicaciones intracraneales; atribuir masa cervical persistente a infección sin descartar neoplasia.",
        "Tratamiento: enfoque escalonado y seguro a nivel pregrado. Incluir medidas generales, analgesia/antiinflamatorios, hidratación, y criterios claros para antibiótico, esteroide (cuando aplique) y referencia. Evitar esquemas avanzados o dosis complejas salvo que se solicite explícitamente.",
        "Seguridad del paciente: NO recomendar automanejo en urgencias ORL; indicar valoración médica/urgencias cuando existan banderas rojas o riesgo de compromiso de vía aérea, sangrado significativo, déficit neurológico, o sospecha tumoral."
    ],
    "urologia": [
        "Generar clases clínicas con razonamiento urológico estructurado: clínica + fisiopatología + interpretación de laboratorio e imagen + conducta.",
        "Abordar siempre desde el síntoma urológico cardinal (disuria, hematuria, dolor lumbar, retención urinaria, síntomas urinarios bajos, masa escrotal, disfunción sexual).",
        "Diferenciar de forma explícita: TAMIZAJE vs DIAGNÓSTICO CONFIRMATORIO vs ESTADIFICACIÓN (cuando aplique).",
        "Integrar laboratorio e imagen de manera racional: EGO, urocultivo, BH, PFH, APE, marcadores tumorales, US, TC, RM y estudios funcionales según contexto clínico.",
        "Incluir siempre diagnóstico diferencial jerarquizado y justificado clínicamente.",
        "Señalar RED FLAGS y criterios de urgencia y referencia: retención aguda de orina, hematuria macroscópica persistente, cólico renal complicado, escroto agudo, torsión testicular, sepsis urinaria, priapismo, síndrome de Fournier.",
        "Tratamiento: enfoque escalonado y seguro a nivel pregrado. Indicar cuándo el manejo es conservador, farmacológico o quirúrgico (sin detallar técnicas avanzadas).",
        "Evitar recomendaciones de automanejo; indicar valoración médica especializada cuando aplique y remarcar urgencias de forma explícita.",
        "No usar 'No aplica'. Si un apartado no es central, adaptarlo al contexto clínico urológico.",
        "Mantener lenguaje clínico claro, preciso y acorde a pregrado; evitar protocolos de alta especialidad."
    ],
    "psiquiatria": [
        "Generar clases clínicas con enfoque psiquiátrico integral: entrevista clínica + psicopatología + diagnóstico sindrómico + diagnóstico diferencial + plan terapéutico inicial.",
        "Siempre iniciar el razonamiento clínico desde la ENTREVISTA PSIQUIÁTRICA y el EXAMEN MENTAL; no emitir diagnósticos sin describir hallazgos psicopatológicos.",
        "Diferenciar explícitamente: síntoma vs signo psicopatológico vs síndrome vs trastorno psiquiátrico.",
        "Usar clasificaciones vigentes DSM-5 y CIE-11 de forma descriptiva, sin memorizar criterios textuales extensos.",
        "En diagnóstico, siempre incluir: diagnóstico principal, diagnósticos diferenciales psiquiátricos y médicos, y comorbilidades frecuentes.",
        "Evaluar de forma sistemática RIESGO SUICIDA, RIESGO HETEROAGRESIVO y CAPACIDAD DE JUICIO cuando el cuadro lo amerite.",
        "En urgencias psiquiátricas, priorizar seguridad del paciente y del entorno antes de cualquier intervención farmacológica.",
        "Tratamiento: dividir siempre en abordaje NO farmacológico y farmacológico; indicar solo esquemas generales (sin dosis) a nivel pregrado.",
        "Señalar criterios claros de REFERENCIA urgente y de hospitalización psiquiátrica cuando existan banderas rojas.",
        "Evitar estigmatización, juicios morales o lenguaje peyorativo; usar terminología clínica profesional.",
        "Diferenciar trastornos primarios psiquiátricos de cuadros secundarios a enfermedades médicas, neurológicas o consumo de sustancias.",
        "Incluir errores clínicos frecuentes en Psiquiatría: sobrediagnóstico, infradiagnóstico, confusión con causas orgánicas, y uso inapropiado de psicofármacos.",
        "No recomendar automanejo; enfatizar seguimiento médico y trabajo multidisciplinario cuando aplique."
    ],
    "medicina_legal": [
        "Enfoque estrictamente médico-legal, académico y descriptivo: definiciones, principios, mecanismos de lesión, documentación y criterios generales de actuación profesional.",
        "Evitar contenido gráfico, morboso o sensacionalista. Usar lenguaje neutral, respetuoso y centrado en dignidad humana.",
        "No incluir instrucciones operativas que puedan facilitar daño, evasión de responsabilidad, manipulación de evidencia o conductas ilegales.",
        "Distinguir siempre: hechos clínicos observables vs inferencias médico-legales vs elementos normativos. Señalar límites de competencia (no es asesoría legal).",
        "Priorizar seguridad del paciente y de terceros: en situaciones de riesgo (violencia, abuso, suicidio, intoxicación) incluir solo conductas generales de referencia institucional y activación de protocolos, sin detallar 'cómo hacerlo'.",
        "En documentación médico-legal: enfatizar registro objetivo (fecha/hora, hallazgos, medición, lenguaje descriptivo, cadena de custodia solo en nivel conceptual).",
        "En violencia sexual y materno-infantil: abordar únicamente conceptos, definiciones, obligaciones de notificación a nivel conceptual y principios de atención centrada en la persona; no detallar técnicas de examen, recolección de evidencia ni procedimientos.",
        "En tanatología: explicar fenómenos cadavéricos y disposiciones mortuorias a nivel conceptual; no describir procedimientos de manipulación del cuerpo o autopsia en forma operativa.",
        "En toxicología: tratar sustancias desde farmacología clínica, toxíndromes, diagnóstico general y criterios de referencia; no incluir formas de consumo, preparación, dosificación o ‘consejos’ que faciliten uso."
    ],
    "nutricion_humana": [
        "Materia PUENTE: integrar nutrición + fisiología/metabolismo + correlación clínica con enfermedad.",
        "Mantener estructura E-Vantis clínica obligatoria (secciones completas) pero con énfasis en evaluación nutricional y toma de decisiones basada en evidencia.",
        "Diferenciar siempre: TAMIZAJE vs EVALUACIÓN nutricional vs DIAGNÓSTICO NUTRICIO vs PLAN/INTERVENCIÓN vs SEGUIMIENTO.",
        "Evitar prescripciones universales: no dietas milagro, no promesas, no recomendaciones absolutas; todo debe ser individualizado al contexto clínico.",
        "Interpretar indicadores por integración: antropométricos, bioquímicos, clínicos, dietéticos y funcionales; no decidir con un solo dato aislado.",
        "Explicar requerimientos energéticos y distribución de macronutrimentos como razonamiento (supuestos y límites), no como números rígidos.",
        "Usar herramientas validadas cuando aplique (VGS/SGA, NRS-2002, MUST, MNA) y explicar qué miden, cuándo usar y limitaciones.",
        "Vincular manejo nutricional con guías clínicas nacionales/internacionales (GPC/NOM cuando aplique) sin citar cifras exactas si no se citará fuente en la misma línea.",
        "Incluir RED FLAGS y criterios de referencia: pérdida ponderal significativa, ingesta insuficiente sostenida, disfagia/aspiración, deshidratación, signos de desnutrición severa, riesgo de síndrome de realimentación, paciente crítico, ERC avanzada, neoplasia activa, VIH avanzado u otras condiciones complejas.",
        "Apoyo nutricional (enteral/parenteral): describir indicaciones/contraindicaciones, riesgos y complicaciones (mecánicas, infecciosas, metabólicas, gastrointestinales) a nivel pregrado, sin protocolos avanzados.",
        "Siempre incluir prevención de errores frecuentes: confundir tamizaje con diagnóstico, usar suplementos sin indicación, restringir de forma excesiva, y omitir seguimiento y reevaluación.",
        "Seguridad del paciente: no recomendar automanejo en condiciones de riesgo; indicar evaluación médica y referencia o urgencias cuando existan banderas rojas."
    ],
    "algologia": [
        "Generar clases clínicas centradas en el razonamiento del dolor: evaluación integral, clasificación fisiopatológica y conducta terapéutica.",
        "Siempre iniciar con definición clara del tipo de dolor y su contexto clínico (agudo, crónico, oncológico, neuropático, total).",
        "Integrar bases anatómicas y neurofisiológicas del dolor (vías nociceptivas, modulación central y plasticidad).",
        "Estructurar la evaluación del dolor con enfoque sistemático: entrevista dirigida, semiología completa y exploración física específica.",
        "Utilizar escalas validadas de medición del dolor según grupo etario y contexto clínico (EVA, ENA, DN4, etc.).",
        "Diferenciar siempre dolor nociceptivo, neuropático y mixto, señalando implicaciones terapéuticas.",
        "Aplicar la escalera analgésica de la OMS y su evolución clínica de forma racional y escalonada.",
        "En el manejo farmacológico, explicar indicaciones, mecanismos, interacciones y principios de prescripción segura.",
        "Identificar RED FLAGS del dolor: dolor desproporcionado, déficit neurológico progresivo, dolor nocturno, fiebre, pérdida ponderal.",
        "Incluir manejo no farmacológico y técnicas intervencionistas cuando el contexto clínico lo amerite.",
        "Abordar el dolor crónico y paliativo desde un enfoque bio-psico-social y multidisciplinario.",
        "Manejar cuidados paliativos desde una perspectiva médica y ética, sin emitir juicios de valor.",
        "Evitar recomendaciones de automanejo; enfatizar referencia oportuna cuando existan criterios de gravedad.",
        "Cerrar cada clase con algoritmos clínicos de evaluación y tratamiento del dolor."
   ],
   "cirugia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones) en TODAS las clases: Definición, Epidemiología y estadística, Cuadro clínico, Signos y síntomas clave, Diagnóstico (enfoque + diferenciales + estándar de oro/tamizaje si aplica), Tratamiento, Pronóstico, Complicaciones, Algoritmos Dx/Tx, Errores frecuentes, Preguntas de repaso.",
        "Enfoque central: razonamiento clínico-quirúrgico (identificar urgencia, priorizar estabilidad, definir conducta: conservador vs quirúrgico, y criterios de referencia/interconsulta).",
        "Siempre diferenciar: URGENCIA vs ELECTIVO; y si aplica, manejo inicial en 1er nivel vs manejo definitivo en 2do/3er nivel.",
        "Diagnóstico: integrar historia y exploración física dirigida + estudios básicos (laboratorio e imagen) con interpretación clínica; incluir diagnósticos diferenciales relevantes y criterios para descartar patología grave.",
        "Siempre incluir RED FLAGS y criterios de inestabilidad: choque, peritonitis, sepsis, hemorragia activa, compromiso de vía aérea, déficit neurológico agudo, isquemia aguda, dolor desproporcionado, fiebre persistente con deterioro, vómito incoercible, datos de perforación/obstrucción.",
        "Tratamiento: describir medidas iniciales seguras (ABC/ATLS cuando aplique, analgesia racional, hidratación, antibióticos cuando estén indicados, control de náusea, reposo digestivo, corrección hidroelectrolítica).",
        "Tratamiento quirúrgico: SOLO generalidades (indicaciones, objetivos, riesgos, contraindicaciones y preparación preoperatoria). PROHIBIDO describir técnicas, pasos operatorios, maniobras, instrumentación o detalles procedimentales.",
        "Siempre incluir: profilaxis y cuidados perioperatorios en generalidades (ayuno, evaluación anestésica, tromboprofilaxis cuando aplique, control glucémico, antibiótico profiláctico si procede) sin esquemas de dosis.",
        "Complicaciones: dividir en tempranas vs tardías cuando aplique; incluir complicaciones de la enfermedad y del tratamiento (médico/quirúrgico) y qué signos obligan revaloración urgente.",
        "Algoritmos: incluir al final un algoritmo práctico de triage y conducta (si estable/inestable; si datos de peritonitis/obstrucción/isquemia; qué estudio pedir primero; cuándo referir/operar).",
        "Mantener lenguaje académico, objetivo y clínico. Evitar contenido gráfico o sensacionalista. No incluir descripciones explícitas de violencia/lesiones; centrarse en mecanismos, hallazgos clínicos y decisiones médicas.",
        "Poblaciones especiales: si aplica, incluir consideraciones breves para embarazo, pediatría, adulto mayor, inmunosupresión y ERC/insuficiencia hepática (sin modificar el temario).",
        "Seguridad del paciente: siempre mencionar consentimiento informado, valoración de riesgo-beneficio y criterios de referencia a Cirugía/urgencias cuando corresponda."
    ],
    "geriatria": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones): definición; epidemiología y estadística; cuadro clínico; signos y síntomas clave; diagnóstico (incluyendo estándar de oro y tamizaje cuando aplique); tratamiento; pronóstico; complicaciones; algoritmos de diagnóstico y tratamiento; preguntas de repaso; errores clínicos frecuentes.",
        "Enfoque geriátrico obligatorio: priorizar funcionalidad, fragilidad, comorbilidad, polifarmacia y contexto sociofamiliar sobre la simple enumeración de enfermedades.",
        "La evaluación clínica debe considerar presentaciones atípicas del adulto mayor (delirium, caídas, inmovilidad, deterioro funcional, anorexia, incontinencia) y no atribuir síntomas únicamente a la edad.",
        "En Diagnóstico, integrar siempre los dominios de la Valoración Geriátrica Integral (VGI): clínico, funcional (AVD/AIVD), cognitivo, afectivo y social; mencionar escalas útiles a nivel pregrado sin puntajes (Katz, Lawton, Tinetti, Mini-Mental o MoCA, Yesavage, Zarit, CAM).",
        "Tamizaje dirigido cuando aplique: fragilidad, riesgo de caídas, delirium, deterioro cognitivo y desnutrición; evitar plantear tamizaje poblacional indiscriminado.",
        "Incluir diagnósticos diferenciales geriátricos clave: delirium vs demencia vs depresión; causas farmacológicas y metabólicas; infecciones con presentación oligosintomática.",
        "Tratamiento: priorizar siempre medidas no farmacológicas (rehabilitación, prevención de caídas, autocuidado, soporte social y educación al cuidador); farmacoterapia solo en generalidades (sin dosis).",
        "Considerar polifarmacia de forma explícita: conciliación de medicamentos, interacciones, ajuste por función renal/hepática y deprescripción cuando corresponda.",
        "Identificar y señalar red flags geriátricas: delirium, caídas recurrentes, síncope, deterioro cognitivo acelerado, maltrato o abandono, desnutrición severa, síndrome de inmovilidad, riesgo suicida, inestabilidad clínica.",
        "Mantener lenguaje clínico profesional, no estigmatizante; evitar términos imprecisos como 'senil'; documentar autonomía, capacidad funcional y objetivos de cuidado.",
        "Cerrar siempre con algoritmos claros y prácticos de diagnóstico y tratamiento adaptados al adulto mayor."
    ],
    "ginecologia_obstetricia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones).",
        "Abordaje integral gineco-obstétrico basado en clínica, fisiopatología y toma de decisiones seguras.",
        "Siempre diferenciar: condición fisiológica vs patológica (especialmente en embarazo, parto y puerperio).",
        "En Diagnóstico: integrar anamnesis dirigida, exploración ginecológica/obstétrica y estudios auxiliares según etapa (sin sobrediagnóstico).",
        "Identificar y señalar RED FLAGS obstétricas y ginecológicas que constituyen urgencia médica o quirúrgica.",
        "Tratamiento: enfoque escalonado, seguro y basado en guías clínicas vigentes; dividir en manejo expectante, médico y quirúrgico cuando aplique.",
        "Evitar indicaciones fuera de contexto gestacional; considerar siempre riesgos maternos y fetales.",
        "Incluir criterios claros de referencia, hospitalización y resolución quirúrgica cuando corresponda.",
        "Mantener lenguaje clínico, ético y no estigmatizante; enfoque médico-científico.",
        "No indicar automanejo; enfatizar seguimiento médico y control prenatal cuando aplique."
    ],
    "infectologia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones).",
        "Abordaje integral infectológico basado en clínica, fisiopatología, epidemiología y toma de decisiones seguras.",
        "Siempre diferenciar: colonización vs infección; infección comunitaria vs nosocomial; y paciente inmunocompetente vs inmunocomprometido.",
        "En Diagnóstico: priorizar razonamiento sindromático (síndrome febril, respiratorio, gastrointestinal, urinario, neurológico, piel/tejidos blandos, sistémico) antes de atribuir etiología; integrar exposiciones, viajes, vacunación y factores de riesgo.",
        "Identificar y señalar RED FLAGS infecciosas (sepsis/choque, alteración del estado mental, dificultad respiratoria, inestabilidad hemodinámica, sospecha de meningitis/encefalitis, neutropenia febril, deshidratación grave, foco profundo).",
        "Tratamiento: enfoque escalonado y seguro; dividir en medidas de soporte, control del foco, y terapia antimicrobiana en generalidades (sin dosis ni esquemas específicos).",
        "Promover uso racional de antimicrobianos y prevención de resistencia: indicar toma de muestras/cultivos y reevaluación clínica para desescalamiento cuando aplique (a nivel conceptual).",
        "Incluir medidas de prevención y control de infecciones (aislamiento, higiene de manos, profilaxis cuando corresponda, vacunación) y criterios de retorno/seguimiento.",
        "Incluir criterios claros de referencia, hospitalización, aislamiento y manejo de urgencias infecciosas cuando corresponda.",
        "Mantener lenguaje clínico, ético y no estigmatizante; enfoque médico-científico.",
        "No indicar automanejo; enfatizar evaluación médica oportuna, seguimiento y vigilancia de complicaciones."
   ],
   "traumatologia_ortopedia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones).",
        "Abordaje clínico-traumatológico y ortopédico basado en exploración física dirigida, mecanismo de lesión y correlación anatómica.",
        "Siempre diferenciar: patología traumática aguda vs patología ortopédica crónica y degenerativa.",
        "En Diagnóstico: priorizar anamnesis orientada a mecanismo de lesión, exploración por segmentos y uso racional de imagen (RX como primera línea; TC/RM según indicación).",
        "Identificar y señalar RED FLAGS traumatológicas: compromiso neurovascular, síndrome compartimental, fractura expuesta, inestabilidad hemodinámica.",
        "Tratamiento: dividir claramente en manejo inicial, conservador y quirúrgico según estabilidad, edad y contexto clínico.",
        "Incluir inmovilización, analgesia, reducción y referencia quirúrgica cuando corresponda.",
        "En población pediátrica: considerar cartílago de crecimiento y clasificaciones específicas (Salter-Harris).",
        "En adulto mayor: integrar fragilidad, osteoporosis y riesgo de caídas en la toma de decisiones.",
        "Indicar criterios claros de referencia urgente, hospitalización y manejo quirúrgico.",
        "Mantener lenguaje clínico técnico, enfoque médico-quirúrgico y sin indicaciones de automanejo."
    ],
    "pediatria": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones).",
        "Enfoque pediátrico obligatorio: integrar edad, peso, percentiles/curvas y etapa del desarrollo en todo el razonamiento clínico.",
        "Siempre considerar contexto familiar/cuidadores, antecedentes perinatales, alimentación y esquema de vacunación cuando aplique.",
        "En Diagnóstico: integrar anamnesis con cuidadores + exploración física completa + signos vitales por edad; evitar sobrediagnóstico.",
        "Identificar y señalar RED FLAGS pediátricas (toxicidad, dificultad respiratoria, alteración del estado de alerta, deshidratación, sepsis, choque).",
        "Diferenciar: fisiológico vs patológico según edad (RN, lactante, escolar, adolescente) y según entorno clínico.",
        "Tratamiento: enfoque escalonado, seguro y basado en clínica; dividir en manejo general, médico y hospitalario/urgente cuando aplique.",
        "No incluir dosis ni esquemas farmacológicos detallados; describir conducta general segura y criterios de referencia.",
        "Incluir criterios claros de referencia, hospitalización y urgencias pediátricas cuando corresponda.",
        "Mantener lenguaje clínico, ético y no estigmatizante; enfatizar seguridad del paciente y seguimiento."
   ],
   "rehabilitacion": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones).",
        "Enfoque funcional y biopsicosocial del paciente.",
        "Integrar diagnóstico, discapacidad y rehabilitación.",
        "No reducir a técnicas aisladas; priorizar razonamiento clínico.",
        "Tratamiento escalonado y centrado en funcionalidad."
   ],
   "reumatologia": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones).",
        "Abordaje sindromático y nosológico de las enfermedades reumatológicas.",
        "Integrar epidemiología, clínica, criterios diagnósticos y razonamiento diferencial.",
        "Priorizar la correlación clínico-inmunológica y fisiopatológica.",
        "Diagnóstico basado en clínica, laboratorio e imagen dirigidos; evitar estudios innecesarios.",
        "Incluir criterios diagnósticos y de clasificación cuando existan.",
        "Tratamiento escalonado según gravedad y actividad de la enfermedad, sin detallar dosis.",
        "Enfatizar identificación de RED FLAGS reumatológicas y criterios de referencia.",
        "Diferenciar claramente procesos inflamatorios, degenerativos y no inflamatorios.",
        "Considerar impacto sistémico, funcional y pronóstico a corto y largo plazo.",
        "Evitar reduccionismo articular; integrar manifestaciones extraarticulares.",
        "Lenguaje académico, clínico y orientado a pregrado."
    ],
    "urgencias": [
        "Materia clínica. Estructura clínica obligatoria completa (11 secciones).",
        "Priorizar seguridad del paciente y estabilización inicial con enfoque ABCDE en todos los temas.",
        "Siempre iniciar con valoración primaria, control de la vía aérea, respiración y circulación según gravedad clínica.",
        "Incluir RED FLAGS y criterios de atención inmediata, hospitalización y referencia a segundo/tercer nivel.",
        "Diagnóstico basado en síndrome y gravedad; solicitar laboratorio e imagen dirigidos, evitando estudios innecesarios.",
        "Tratamiento escalonado por prioridades vitales; NO incluir dosis ni esquemas farmacológicos detallados.",
        "Incluir medidas no farmacológicas y de soporte (oxigenación, fluidos, monitoreo, accesos) cuando aplique.",
        "Integrar diagnósticos diferenciales críticos que no deben omitirse (amenazas vitales).",
        "Enfatizar criterios de alta segura y de revaloración, cuando el cuadro lo permita.",
        "Incluir algoritmos claros de diagnóstico y tratamiento con toma de decisiones rápida.",
        "Evitar recomendaciones de automanejo; enfatizar evaluación médica y seguimiento."
    ],
}

# ----------------------------
# Chat memory (tu implementación)
# ----------------------------
MAX_TURNS_RAW = 6
SUMMARIZE_AFTER_TURNS = 8
SUMMARY_MAX_TOKENS = 250
SUMMARY_MODEL = os.getenv("EVANTIS_SUMMARY_MODEL", "gpt-4.1-mini")

BASE_SYSTEM = """
Eres E-VANTIS, un asistente médico-académico de alta exigencia.

OBJETIVO:
Entregar respuestas profundas, estructuradas, didácticas y clínicamente seguras cuando aplique.

REGLAS DE ESTILO (SIEMPRE):
1) No mezcles temas en una misma respuesta.
2) Estructura obligatoria con encabezados claros.
3) Define términos clave antes de usarlos si pueden ser ambiguos.
4) Usa lenguaje médico profesional, con claridad docente.
5) Incluye perlas/high-yield cuando aplique.
""".strip()

MODE_PROMPTS = {
    "academico": """
MODO: ACADÉMICO (E-Vantis).
Objetivo: aprender y comprender.
- Prioriza definiciones formales, clasificaciones, fisiopatología y mecanismos.
- Explica el “por qué” antes del “qué hacer”.
- Incluye trampas/perlas TEÓRICAS cuando aplique.
""".strip(),
    "clinico": """
MODO: CLÍNICO (E-Vantis).
Objetivo: decidir y actuar.
- Prioriza presentación clínica, signos/síntomas clave, diagnóstico diferencial, red flags.
- Enfatiza enfoque diagnóstico y conducta inicial/algoritmos.
- Incluye errores clínicos frecuentes y perlas tipo ENARM/USMLE.
""".strip(),
}

MODEL_BY_MODE = {
    "academico": os.getenv("EVANTIS_OPENAI_MODEL", "gpt-4.1-mini"),
    "clinico": os.getenv("EVANTIS_OPENAI_MODEL", "gpt-4.1-mini"),
}

DETAIL_MAX_TOKENS = {
    "breve": 350,
    "extendido": 900,
    "maximo": 1400,
}

SUMMARY_PROMPT = """
Eres un módulo de memoria de E-VANTIS.
Crea un resumen breve y útil del contexto acumulado para mantener continuidad.
Formato:
- Tema principal
- Objetivo del usuario
- Decisiones/criterios acordados
- Pendientes
No inventes información.
""".strip()

CACHE: Dict[str, Dict[str, object]] = {}


def get_session(session_id: str):
    if session_id in CACHE:
        return CACHE[session_id]["summary"], CACHE[session_id]["history"]
    summary, history = db_load_session(session_id)
    CACHE[session_id] = {"summary": summary, "history": history}
    return summary, history


def set_session(session_id: str, summary: str, history: List[Dict[str, str]]):
    CACHE[session_id] = {"summary": summary, "history": history}
    db_save_session(session_id, summary, history)


def summarize_if_needed(session_id: str, summary: str, history: List[Dict[str, str]]):
    turns = len(history) // 2
    if turns <= SUMMARIZE_AFTER_TURNS:
        return summary, history

    input_msgs = [{"role": "system", "content": SUMMARY_PROMPT}]
    if summary:
        input_msgs.append({"role": "user", "content": f"RESUMEN PREVIO:\n{summary}"})
    input_msgs.append({"role": "user", "content": f"HISTORIAL:\n{json.dumps(history, ensure_ascii=False)}"})

    resp = client.responses.create(
        model=SUMMARY_MODEL,
        input=input_msgs,
        max_output_tokens=SUMMARY_MAX_TOKENS,
    )
    new_summary = (resp.output_text or "").strip()
    kept = history[-(MAX_TURNS_RAW * 2):]
    return new_summary, kept


# ----------------------------
# Requests / Responses
# ----------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Identificador de sesión (persistente).")
    message: str = Field(..., description="Mensaje del usuario.")
    mode: Literal["academico", "clinico"] = "academico"
    detail_level: Literal["breve", "extendido", "maximo"] = "extendido"
    hard_max_tokens: Optional[int] = Field(None, description="Sobrescribe el máximo de tokens de salida.")


class ChatResponse(BaseModel):
    session_id: str
    mode: str
    detail_level: str
    response: str
    used_summary: bool


class TeachRequest(BaseModel):
    session_id: str = Field(..., description="Sesión persistente (se guarda en SQLite).")
    topic: str = Field(..., description="Tema de la clase.")
    mode: Literal["academico", "clinico"] = "clinico"
    level: Level = "auto"
    duration_minutes: int = Field(20, ge=5, le=60, description="Duración objetivo de la clase.")
    style: Literal["magistral", "high_yield", "socratico"] = "magistral"


class TeachResponse(BaseModel):
    session_id: str
    subject: str
    level: str
    duration_minutes: int
    title: str
    lesson: str


class RegisterRequest(BaseModel):
    email: str
    password: str

class RegisterResponse(BaseModel):
    ok: bool = True
    status: Literal["pending_verification", "registered"] = "pending_verification"
    email: str
    verify_link: Optional[str] = None  # solo para MVP/logs

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    plan: str


class CreateKeyRequest(BaseModel):
    role: Literal["student", "admin"] = "student"
    label: str = "Alumno"
    api_key: Optional[str] = None


class CreateKeyResponse(BaseModel):
    api_key: str
    role: str
    label: str
    is_active: bool


class TeachCurriculumIn(BaseModel):
    subject_id: str
    topic_id: str
    subtopic_id: Optional[str] = None

    level: Optional[str] = "auto"
    duration_minutes: Optional[int] = 20
    mode: Optional[Literal["academico", "clinico"]] = "clinico"
    style: Optional[str] = "magistral"

    study_mode: StudyMode = "clinico"
    module: Module = "lesson"

    enarm_context: StrictBool = False
    use_guides: StrictBool = False

    num_questions: int = 8

    editorial_v1: StrictBool = False


# ----------------------------
# Helpers: curriculum, slug, subtopics
# ----------------------------
def slugify_subtopic(text: str) -> str:
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text[:80] or "subtopic"


def normalize_subtopics(subtopics):
    if not subtopics:
        return []
    normalized = []
    for st in subtopics:
        if isinstance(st, dict):
            normalized.append(
                {
                    "id": st.get("id") or slugify_subtopic(st.get("name", "subtopic")),
                    "name": st.get("name") or st.get("id") or "Subtópico",
                }
            )
        elif isinstance(st, str):
            normalized.append({"id": slugify_subtopic(st), "name": st})
        else:
            normalized.append({"id": "subtopic", "name": str(st)})

    seen = set()
    dedup = []
    for st in normalized:
        sid = st["id"]
        if sid in seen:
            base = sid
            i = 2
            while f"{base}_{i}" in seen:
                i += 1
            st = {**st, "id": f"{base}_{i}"}
        seen.add(st["id"])
        dedup.append(st)
    return dedup


def detect_level_simple(topic: str) -> str:
    t = (topic or "").lower()
    internado_keywords = [
        "manejo", "tratamiento", "conducta", "algoritmo", "abordaje", "urgencias",
        "choque", "shock", "sepsis", "paro", "rcp", "intub", "vía aérea", "via aerea",
        "dosis", "mg", "ml", "infusión", "infusion",
        "ekg", "electro",
    ]
    return "internado" if any(k in t for k in internado_keywords) else "pregrado"


def usage_monthly_get_all(conn, user_id: str, yyyymm: str) -> dict:
    cur = conn.execute(
        "SELECT module, count FROM usage_monthly WHERE user_id=? AND yyyymm=?",
        (user_id, yyyymm),
    )
    rows = cur.fetchall() or []
    return {m: int(c) for (m, c) in rows}


def build_usage_snapshot(plan: str, used_by_module: dict) -> dict:
    plan = (plan or "free").lower()
    snapshot = {}
    for module in ("lesson", "exam", "enarm", "gpc_summary"):
        limit = _quota_limit(plan, module)
        used = int(used_by_module.get(module, 0))
        remaining = max(0, int(limit) - used)
        snapshot[module] = {
            "used": used,
            "limit": limit,
            "remaining": remaining,
            "blocked": (limit == 0),
        }
    return snapshot


# ----------------------------
# Module prompting (EXAM)
# ----------------------------
def _exam_spec_for_study_mode(study_mode: StudyMode) -> str:
    if study_mode == "basico":
        return "Examen conceptual (mecanismos, definiciones, clasificación). SIN casos clínicos ni conducta."
    if study_mode == "clinico":
        return "Examen con enfoque diagnóstico: signos clave, diferenciales razonados, pruebas indicadas."
    if study_mode == "internado":
        return "Examen prioriza conducta/urgencias: triage, ABC, decisiones iniciales, red flags."
    if study_mode == "examen":
        return "Examen discriminativo estilo ENARM: preguntas retadoras, trampas comunes, elección de la mejor respuesta."
    return "Examen de medicina (general)."


def build_exam_prompt(
    subject_name: str,
    topic_name: str,
    level: str,
    study_mode: StudyMode,
    n_questions: int = 15,
) -> str:
    spec = _exam_spec_for_study_mode(study_mode)
    return f"""
Genera un examen de opción múltiple.

Materia: {subject_name}
Tema: {topic_name}
Nivel: {level}
Tipo: {study_mode}

Requisitos:
- {spec}
- {n_questions} preguntas.
- Formato por pregunta:
  Q#) Enunciado
  A) ...
  B) ...
  C) ...
  D) ...
  Respuesta correcta: <letra>
  Explicación breve (2–5 líneas) justificada por razonamiento clínico-académico.
- Prohibido copiar reactivos oficiales o texto de guías. Crea casos y enunciados originales.
- Al final incluye 5 “Errores típicos” (sesgos o confusiones).
""".strip()


# ----------------------------
# Health/ready
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ready")
def ready():
    if not os.getenv("EVANTIS_JWT_SECRET"):
        raise HTTPException(status_code=500, detail="JWT secret no configurado")
    try:
        with db_conn() as conn:
            conn.execute("SELECT 1")
    except Exception:
        raise HTTPException(status_code=500, detail="DB no disponible")
    return {"ready": True}


# ----------------------------
# Chat
# ----------------------------
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit_chat)])
def chat(req: ChatRequest, user: dict = Depends(require_user)):
    if req.mode not in MODE_PROMPTS:
        raise HTTPException(status_code=422, detail="Modo inválido: Solo: academico, clinico.")

    plan = (user.get("plan") or "free").strip().lower()
    enforce_chat_quota(user["user_id"], plan)

    summary, history = get_session(req.session_id)

    system_prompt = "\n\n".join([BASE_SYSTEM, MODE_PROMPTS[req.mode]]).strip()

    max_tokens = req.hard_max_tokens if req.hard_max_tokens is not None else DETAIL_MAX_TOKENS[req.detail_level]
    model = MODEL_BY_MODE.get(req.mode, os.getenv("EVANTIS_OPENAI_MODEL", "gpt-4.1-mini"))

    messages = [{"role": "system", "content": system_prompt}]
    used_summary = False
    if summary:
        used_summary = True
        messages.append({"role": "system", "content": f"MEMORIA RESUMIDA DE LA SESIÓN:\n{summary}"})
    messages.extend(history)
    messages.append({"role": "user", "content": req.message})

    response = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
    )
    answer = (response.output_text or "").strip()

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})

    summary, history = summarize_if_needed(req.session_id, summary, history)
    set_session(req.session_id, summary, history)

    db_log_usage(
        user_id=user["user_id"],
        endpoint="/chat",
        subject_id=None,
        topic_id=None,
        module="chat",
        used_web_search=False,
        model=model,
        approx_output_chars=len(answer),
    )

    consume_chat_quota(user["user_id"])

    return ChatResponse(
        session_id=req.session_id,
        mode=req.mode,
        detail_level=req.detail_level,
        response=answer,
        used_summary=used_summary,
    )


# ----------------------------
# Auth
# ----------------------------
@app.get("/auth/verify-email")
def verify_email(token: str):
    if not token or len(token) < 16:
        raise HTTPException(status_code=400, detail="Token inválido.")

    row = db_get_user_by_verify_token(token)
    if not row:
        raise HTTPException(status_code=404, detail="Token no encontrado o ya usado.")

    if bool(row.get("email_verified", False)):
        return {"ok": True, "status": "already_verified"}

    expires_at = int(row.get("expires_at") or 0)
    if expires_at and int(time.time()) > expires_at:
        raise HTTPException(status_code=410, detail="Token expirado. Solicita uno nuevo.")

    db_mark_email_verified(row["user_id"])
    return {"ok": True, "status": "verified"}

@app.post("/auth/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
    email = (req.email or "").strip().lower()

    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Email inválido.")
    if not req.password or len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password muy corto (mínimo 8).")
    if len(req.password.encode("utf-8")) > 128:
        raise HTTPException(status_code=400, detail="Password demasiado largo (máximo 128 bytes).")

    if db_get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Ese email ya está registrado.")

    try:
        pw_hash = hash_password(req.password)
    except ValueError:
        raise HTTPException(status_code=400, detail="Password inválido. Revisa longitud y caracteres.")
    except Exception:
        raise HTTPException(status_code=500, detail="No se pudo procesar la contraseña. Intenta de nuevo.")

    user_id = db_create_user(email=email, password_hash=pw_hash, plan="free")

    # A3: generar token verify
    if EVANTIS_EMAIL_VERIFY_ENABLED:
        token = "evv_" + secrets.token_urlsafe(24)
        expires_at = int(time.time()) + EVANTIS_EMAIL_VERIFY_TTL_SECONDS
        db_set_email_verification(user_id=user_id, token=token, expires_at=expires_at)
        verify_link = send_verify_email(email=email, token=token)

        # Por seguridad: NO devolver verify_link en prod.
        # Si necesitas QA rápido, set EVANTIS_RETURN_VERIFY_LINK=1
        return RegisterResponse(
            ok=True,
            status="pending_verification",
            email=email,
            verify_link=verify_link if EVANTIS_RETURN_VERIFY_LINK else None,
        )


    # Si desactivas email verify (solo para entornos cerrados)
    db_mark_email_verified(user_id)
    return RegisterResponse(ok=True, status="registered", email=email, verify_link=None)

@app.post("/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), request: Request = None):
    email = (form_data.username or "").strip().lower()
    password = form_data.password or ""

    if not email or "@" not in email:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    if not password:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    if len(password.encode("utf-8")) > 128:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    user = db_get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    stored = user.get("password_hash", "") or ""

    if is_sha256_hex(stored):
        if not verify_password_sha256(password, stored):
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        try:
            new_hash = hash_password(password)
            db_update_password_hash(user["user_id"], new_hash)
            stored = new_hash
        except Exception:
            pass
    else:
        if not verify_password(password, stored):
            raise HTTPException(status_code=401, detail="Credenciales inválidas")

    try:
        db_touch_login(user["user_id"])
    except Exception:
        pass

    # A3: bloquear login hasta verificar email (excepto QA)
    if (
        EVANTIS_EMAIL_VERIFY_ENABLED
        and not EVANTIS_ALLOW_UNVERIFIED_LOGIN
        and not bool(user.get("email_verified", False))
    ):
        raise HTTPException(
            status_code=403,
            detail="Correo no verificado. Revisa tu bandeja o solicita reenvío del enlace.",
        )

    plan = (user.get("plan") or "free").strip().lower()
    if plan not in ("free", "pro", "premium"):
        plan = "free"

    sid = "sid_" + uuid.uuid4().hex
    jti = "jti_" + uuid.uuid4().hex
    now = int(time.time())
    exp = now + (JWT_EXPIRE_MIN * 60)

    db_revoke_user_sessions(user["user_id"])

    ip = request.client.host if request and request.client else None
    ua = request.headers.get("user-agent") if request else None
    db_insert_session(user["user_id"], sid, jti, now, exp, ip=ip, user_agent=ua)

    token = create_access_token(user_id=user["user_id"], plan=plan, sid=sid, jti=jti)  # type: ignore[arg-type]
    return TokenResponse(access_token=token, token_type="bearer", plan=plan)


# ----------------------------
# Admin keys
# ----------------------------
@app.post("/admin/keys", response_model=CreateKeyResponse, dependencies=[Depends(require_admin)])
def admin_create_key(req: CreateKeyRequest):
    api_key = req.api_key or f"ev_{secrets.token_urlsafe(24)}"
    db_create_key(api_key=api_key, role=req.role, label=req.label)
    return CreateKeyResponse(api_key=api_key, role=req.role, label=req.label, is_active=True)


@app.get("/admin/keys", dependencies=[Depends(require_admin)])
def admin_list_keys():
    rows = db_list_keys()
    out = []
    for api_key, role, label, is_active, created_at, last_used_at in rows:
        out.append(
            {
                "api_key": api_key,
                "role": role,
                "label": label,
                "is_active": bool(int(is_active)),
                "created_at": created_at,
                "last_used_at": last_used_at,
            }
        )
    return {"keys": out}


@app.post("/admin/keys/revoke", dependencies=[Depends(require_admin)])
def admin_revoke_key(api_key: str):
    db_revoke_key(api_key)
    return {"ok": True, "api_key": api_key}


@app.post("/reset", dependencies=[Depends(require_admin)])
def reset_session(session_id: str):
    CACHE.pop(session_id, None)
    db_delete_session(session_id)
    return {"ok": True, "session_id": session_id}


@app.get("/session", dependencies=[Depends(require_admin)])
def get_session_debug(session_id: str):
    summary, history = get_session(session_id)
    return {"session_id": session_id, "summary": summary, "history": history, "db_path": DB_PATH}


@app.get("/usage/me")
def usage_me(user: dict = Depends(require_user)):
    conn = db_conn()
    try:
        yyyymm = _yyyymm_utc()
        plan = (user.get("plan") or "free").strip().lower()

        used_by_module = usage_monthly_get_all(conn, user["user_id"], yyyymm)
        modules = build_usage_snapshot(plan, used_by_module)

        return {
            "user_id": user["user_id"],
            "plan": plan,
            "yyyymm": yyyymm,
            "modules": modules,
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ----------------------------
# /teach (general) — PATCH: prompt limpio + TeachResponse correcto
# ----------------------------
@app.post("/teach", response_model=TeachResponse, dependencies=[Depends(rate_limit_teach)])
def teach(req: TeachRequest, response: Response, claims: dict = Depends(require_user)):
    level = req.level if req.level != "auto" else detect_level_simple(req.topic)
    mode = "clinico"
    model = MODEL_BY_MODE.get(mode, os.getenv("EVANTIS_OPENAI_MODEL", "gpt-4.1-mini"))

    system_msg = """
Eres E-VANTIS.
Actúas como profesor universitario de medicina con alta exigencia académica.
Redactas en español, con precisión terminológica y estructura clara.
""".strip()

    user_msg = f"""
Genera una clase clínica completa.

Tema:
{req.topic}

Nivel:
{level}

Duración:
{req.duration_minutes} minutos

{build_evantis_header_instruction("General", level, req.duration_minutes, req.style)}

{build_clinical_template_instruction()}
""".strip()

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content_text = (resp.output_text or "").strip()
    used_web_search = _resp_used_web_search(resp)

    db_log_usage(
        user_id=claims["user_id"],
        endpoint="/teach",
        subject_id=None,
        topic_id=None,
        module="lesson",
        used_web_search=used_web_search,
        model=model,
        approx_output_chars=len(content_text),
    )

    return TeachResponse(
        session_id=req.session_id,
        subject="general",
        level=level,
        duration_minutes=req.duration_minutes,
        title=req.topic,
        lesson=content_text,
    )


def as_str(x: Any) -> str:
    return (getattr(x, "value", x) or "").strip()


# ----------------------------
# /teach/curriculum — módulos + guías bajo demanda
# ----------------------------
@app.post("/teach/curriculum", dependencies=[Depends(rate_limit_teach)])
def teach_curriculum(
    payload: TeachCurriculumIn,
    request: Request,
    user: dict = Depends(require_user),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    conn = db_conn()
    ip = request.client.host if request.client else "unknown"
    enforce_rate_limit(conn, user["user_id"], ip, "/teach/curriculum", limit_per_minute=30)
    if idempotency_key:
        enforce_idempotency(conn, user["user_id"], idempotency_key, ttl_seconds=30)
    try:
        duration = int(payload.duration_minutes or 20)
        style = (payload.style or "magistral").strip()
        level = (payload.level or "auto").strip()

        if duration < 5 or duration > 60:
            raise HTTPException(status_code=422, detail="duration_minutes debe estar entre 5 y 60.")

        if level.lower() == "clinico":
            level = "internado"

        module = (as_str(payload.module) or "lesson").strip().lower()
        if module not in ("lesson", "exam", "enarm", "gpc_summary"):
            raise HTTPException(status_code=422, detail=f"Módulo no soportado: {module}")

        study_mode = (as_str(payload.study_mode) or "clinico").strip().lower()
        use_guides = bool(payload.use_guides or False)
        plan = (user.get("plan") or "free").strip().lower()
        subject_id = (as_str(payload.subject_id) or "").strip().lower()
        raw_topic_id = (as_str(payload.topic_id) or "").strip()
        editorial_v1 = bool(getattr(payload, "editorial_v1", False))

        if "::" in raw_topic_id:
            topic_id, subtopic_id = raw_topic_id.split("::", 1)
            topic_id = topic_id.strip()
            subtopic_id = subtopic_id.strip()
        else:
            topic_id = raw_topic_id
            subtopic_id = None

        # ----------------------------
        # Plan gating
        # ----------------------------
        if module == "enarm" and not can_use_enarm(plan):
            raise HTTPException(status_code=403, detail="ENARM solo disponible en Pro/Premium.")
        if module == "gpc_summary" and not can_use_gpc_summary(plan):
            raise HTTPException(status_code=403, detail="Resumen GPC solo disponible en Pro/Premium.")
        if use_guides and not can_use_web_search(plan):
            raise HTTPException(status_code=403, detail="Uso de guías actualizadas (web search) solo en Pro/Premium.")

        # ----------------------------
        # FASE 8 — Cuotas mensuales (ANTES de generar)
        # ----------------------------
        yyyymm = _yyyymm_utc()
        limit = _quota_limit(plan, module)

        if limit == 0:
            raise HTTPException(status_code=429, detail="Este módulo no está disponible en tu plan.")

        current_count = usage_monthly_get_count(conn, user["user_id"], module, yyyymm)
        if current_count >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Límite mensual alcanzado para {module} ({limit}/mes)."
            )

        # ----------------------------
        # Cargar curriculum
        # ----------------------------
        curriculum_path = BASE_DIR / "curriculum" / "evantis.curriculum.v1.json"
        curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))

        subject = next((s for s in curriculum.get("subjects", []) if s.get("id") == subject_id), None)
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")

        subject_name = subject.get("name") or payload.subject_id

        if level == "auto":
            level = subject.get("level_default") or "pregrado"

        npm_profile = (subject.get("npm_profile") or "").lower()
        PROFILE_ALIASES = {"clinico": "clinicas", "basica": "basicas"}
        npm_profile = PROFILE_ALIASES.get(npm_profile, npm_profile)

        if npm_profile not in ("basicas", "puente", "clinicas"):
            raise HTTPException(status_code=500, detail=f"NPM profile no reconocido: {npm_profile}")

        # ----------------------------
        # Encontrar topic (macro_topic)
        # ----------------------------
        topic = None
        for block in subject.get("blocks", []):
            for macro in block.get("macro_topics", []):
                if (macro.get("id") or "").strip() == topic_id:
                    topic = macro
                    break
            if topic:
                break

        if not topic:
            raise HTTPException(
                status_code=404,
                detail=f"Topic not found: requested={repr(topic_id)} subject={repr(subject_id)}"
            )

        topic_name = topic.get("name") or payload.topic_id

        # ----------------------------
        # Subtópicos + NPM del tema
        # ----------------------------
        subtopics = normalize_subtopics(topic.get("subtopics", []) or [])
        subtopics_text = "\n".join(f"- {st['name']}" for st in subtopics) if subtopics else "- (Sin subtópicos)"

        topic_npm = topic.get("npm_rules", []) or []
        topic_npm_text = "\n".join([f"- {r}" for r in topic_npm]) if topic_npm else "- (Sin NPM por tema)"

        # ----------------------------
        # Reglas del sistema
        # ----------------------------
        system_rules = build_system_npm(npm_profile)
        system_rules_text = "\n".join([f"- {r}" for r in system_rules]) if system_rules else "- (Sin reglas sistema)"

        subject_rules = NPM_SUBJECT_RULES.get(subject_id, [])
        subject_rules_text = "\n".join([f"- {r}" for r in subject_rules]) if subject_rules else ""

        system_msg = f"""
Eres E-VANTIS.
Profesor universitario de medicina en la materia: {subject_name}.
""".strip()

        tools = []
        used_web_search = False

        # ======================================================================
        # MODULE: LESSON
        # ======================================================================
        if module == "lesson":
            user_msg = f"""
Solicitud: lesson

Materia: {subject_name}
Tema: {topic_name}
Nivel: {level}
Perfil NPM: {npm_profile}

NPM SISTEMA (OBLIGATORIAS: BASE + PERFIL):
{system_rules_text}

NPM DEL TEMA (OBLIGATORIAS):
{topic_npm_text}

Subtópicos:
{subtopics_text}
""".strip()

            if subject_rules_text:
                user_msg += f"\n\nReglas específicas de la materia (OBLIGATORIAS):\n{subject_rules_text}"

            if npm_profile == "clinicas":
                lvl = (level or "auto").strip().lower()

                user_msg += """

# REGLAS E-VANTIS — PROFUNDIDAD POR NIVEL (CLÍNICAS)
Regla madre: En materias clínicas NO se omiten secciones ni se altera el orden. Ajusta únicamente la profundidad del contenido según el nivel.

Cumplimiento mínimo obligatorio:
- Mantén EXACTAMENTE las 11 secciones clínicas (H2) en orden, sin secciones extra.
- En "Diagnóstico" incluye SIEMPRE:
  1) Enfoque diagnóstico
  2) Diagnósticos diferenciales (3–6) con una línea discriminativa
  3) Estándar de oro (nombrar)
  4) Tamizaje (si aplica; si no aplica, redacción equivalente útil)
- En "Tratamiento" incluye SIEMPRE:
  objetivos, medidas generales seguras, primera línea por principios/familias (sin dosis ni esquemas),
  y criterios de referencia/urgencia.
- En algoritmos incluye SIEMPRE:
  qué evaluar primero, red flags y cuándo referir.
- Prohibido usar “No aplica” como respuesta aislada.
- Prohibido incluir dosis, esquemas o protocolos avanzados salvo solicitud explícita.
"""

                if lvl == "pregrado":
                    user_msg += """

Nivel PREGRADO:
- Diagnóstico general y conceptual.
- Tratamiento por principios y medidas seguras.
- Algoritmos básicos con red flags y criterios de referencia.
"""
                elif lvl == "internado":
                    user_msg += """

Nivel CLÍNICO (equivalente a internado):
- Diagnóstico operativo con criterios de gravedad.
- Tratamiento con conducta inicial + escalamiento + referencia.
- Algoritmos orientados a decisión clínica.
"""
                else:
                    user_msg += """

Nivel AUTO:
- Comportarse como PREGRADO.
"""

            if editorial_v1:
                user_msg += "\n\n" + PHASE4_MD_CONVENTION_V1
                user_msg += "\n\nInstrucción: usa badges/callouts SOLO cuando realmente aporten (no saturar)."

            user_msg += "\n\n" + build_evantis_header_instruction(subject_name, level, duration, style)

            if npm_profile == "basicas":
                user_msg += "\n\n" + build_basic_template_instruction()
            elif npm_profile == "puente":
                user_msg += "\n\n" + build_bridge_template_instruction()
            else:
                user_msg += "\n\n" + build_clinical_template_instruction()

            # PATCH: NO forces un segundo H2 "Preguntas de repaso" en clínicas.
            # En clínicas YA viene en CLINICAL_SECTIONS y tu validador exige headings exactos.
            if npm_profile != "clinicas":
                user_msg += """

# BLOQUE OBLIGATORIO AL FINAL (SIEMPRE)
Al final del documento agrega EXACTAMENTE este encabezado H2:
## Preguntas de repaso

Debajo incluye 5–8 preguntas numeradas (1., 2., 3., ...) basadas en el contenido.
No omitir este bloque.
""".strip()

        # ======================================================================
        # MODULE: EXAM
        # ======================================================================
        elif module == "exam":
            user_msg = build_exam_prompt(subject_name, topic_name, level, study_mode, n_questions=15)

        # ======================================================================
        # MODULE: ENARM
        # ======================================================================
        elif module == "enarm":
            if npm_profile == "basicas":
                raise HTTPException(status_code=422, detail="ENARM no disponible en materias básicas.")
            if not payload.enarm_context:
                raise HTTPException(status_code=422, detail="Para ENARM activa enarm_context=true.")

            n = int(getattr(payload, "num_questions", 8) or 8)
            if n < 4 or n > 20:
                raise HTTPException(status_code=422, detail="num_questions debe estar entre 4 y 20.")

            user_msg = f"""
Genera un CASO CLÍNICO SERIADO estilo ENARM sobre:
Materia: {subject_name}
Tema: {topic_name}
Nivel: {level}

Requisitos:
- Caso clínico completo.
- 1 caso con {n} preguntas seriadas tipo ENARM.
- Enfoque discriminativo diagnóstico/interpretación/conducta.
- NO incluyas dosis exactas salvo que el usuario lo pida.
""".strip()

            if use_guides:
                tools = [{"type": "web_search"}]
                user_msg += "\n\nUsa web_search para verificar vigencia (GPC México), sin copiar texto literal."

            system_msg = """
Eres E-VANTIS.
Especialista en preparación ENARM.
""".strip()

        # ======================================================================
        # MODULE: GPC SUMMARY
        # ======================================================================
        elif module == "gpc_summary":
            if npm_profile == "basicas":
                raise HTTPException(status_code=422, detail="gpc_summary no disponible en materias básicas.")

            user_msg = build_gpc_summary_prompt(subject_name, topic_name)
            tools = [{"type": "web_search"}]
            system_msg = "Eres E-VANTIS. Especialista en GPC mexicanas."

        else:
            raise HTTPException(status_code=422, detail=f"Módulo no soportado: {module}")

        # ----------------------------
        # OpenAI
        # ----------------------------
        model = os.getenv("EVANTIS_OPENAI_MODEL", "gpt-4.1-mini")

        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            tools=tools or [],
        )

        content_text = (resp.output_text or "").strip()
        used_web_search = _resp_used_web_search(resp)
        if not content_text:
            raise HTTPException(status_code=502, detail="Modelo devolvió contenido vacío.")

        # ----------------------------
        # Validaciones
        # ----------------------------
        if module == "lesson" and npm_profile == "clinicas":
            ok_struct, errs_struct = validate_clinical_markdown(content_text)
            attempts = 1
            while (not ok_struct) and attempts < 3:
                repair_msg = build_repair_instruction(errs_struct) + "\n\n---\n\n" + content_text
                resp2 = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": repair_msg},
                    ],
                )
                content_text = (resp2.output_text or "").strip()
                ok_struct, errs_struct = validate_clinical_markdown(content_text)
                attempts += 1

            if not ok_struct:
                raise HTTPException(
                    status_code=500,
                    detail="Clase inválida (no cumple estándar clínico E-Vantis): " + " | ".join(errs_struct),
                )

        if module == "gpc_summary":
            ok_gpc, errs_gpc = validate_gpc_summary(content_text)
            attempts = 1

            while (not ok_gpc) and attempts < 3:
                repair_msg = (
                    "Tu respuesta NO cumple el estándar E-VANTIS para resúmenes GPC.\n\n"
                    "Errores detectados:\n- " + "\n- ".join(errs_gpc) + "\n\n"
                    "Corrige y devuelve SOLO el Markdown final cumpliendo TODAS las reglas.\n\n"
                    "Reglas obligatorias:\n"
                    "- Mantén EXACTAMENTE la sección: '## Validación de la GPC consultada'\n"
                    "- Dentro de esa sección incluye EXACTAMENTE estas líneas:\n"
                    "  - Nombre:\n"
                    "  - Año:\n"
                    "  - Institución:\n"
                    "  - Última actualización:\n"
                    "  - Enlace:\n"
                    "- Si no existe fecha de actualización, escribe:\n"
                    "  'Última actualización: no especificada en la fuente consultada.'\n"
                    "- NO inventes enlaces. Si no hay URL exacta:\n"
                    "  'Enlace: no disponible en la consulta.'\n"
                )

                resp2 = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": repair_msg + "\n\n---\n\n" + content_text},
                    ],
                    tools=[{"type": "web_search"}],
                )

                content_text = (resp2.output_text or "").strip()
                ok_gpc, errs_gpc = validate_gpc_summary(content_text)
                attempts += 1

            if not ok_gpc:
                raise HTTPException(
                    status_code=500,
                    detail="Resumen GPC inválido: " + " | ".join(errs_gpc),
                )

        # Validación estándar E-Vantis: Preguntas de repaso (todas las lessons)
        if module == "lesson":
            if not has_review_questions(content_text):
                repair_msg = """
FALTA UN REQUISITO OBLIGATORIO DEL ESTÁNDAR E-VANTIS.

Agrega AL FINAL del documento EXACTAMENTE:
## Preguntas de repaso

Incluye 5–8 preguntas numeradas (1., 2., 3., ...) basadas en el contenido.
NO modifiques el resto del documento; solo añade el bloque final.
""".strip() + "\n\n---\n\n" + content_text

                resp_fix = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": repair_msg},
                    ],
                    tools=tools or [],
                )

                content_text = (resp_fix.output_text or "").strip()
                if not content_text:
                    raise HTTPException(status_code=502, detail="Modelo devolvió contenido vacío (repair preguntas de repaso).")

                if not has_review_questions(content_text):
                    raise HTTPException(status_code=500, detail="Clase inválida: faltan Preguntas de repaso al final.")

        # --------------------------------------------------
        # FASE 8 — incrementar cuota SOLO en éxito
        # --------------------------------------------------
        usage_monthly_increment(conn, user["user_id"], module, yyyymm)
        conn.commit()

        # ----------------------------
        # Logging + response
        # ----------------------------
        db_log_usage(
            user_id=user["user_id"],
            endpoint="/teach/curriculum",
            subject_id=payload.subject_id,
            topic_id=payload.topic_id,
            module=module,
            used_web_search=used_web_search,
            model=model,
            approx_output_chars=len(content_text),
        )

        resp_out = {
            "subject": subject_name,
            "subject_id": payload.subject_id,
            "topic": topic_name,
            "topic_id": payload.topic_id,
            "level": level,
            "duration_minutes": duration,
            "module": module,
            "study_mode": study_mode,
            "used_guides": bool(used_web_search),
            "certifiable": (module != "gpc_summary") or bool(used_web_search),
            "plan": plan,
            "editorial_v1": editorial_v1,
        }

        if module == "lesson":
            resp_out["lesson"] = content_text
        elif module == "exam":
            resp_out["exam"] = content_text
        elif module == "enarm":
            resp_out["enarm"] = content_text
        elif module == "gpc_summary":
            resp_out["gpc_summary"] = content_text
        else:
            resp_out["content"] = content_text

        return resp_out

    finally:
        try:
            conn.close()
        except Exception:
            pass