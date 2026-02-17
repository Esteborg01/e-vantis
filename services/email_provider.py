import os
import json
import requests
from typing import Optional, Dict, Any

# Resend API
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "").strip()
EMAIL_FROM = os.getenv("EMAIL_FROM", "").strip()

# Base URL del frontend para links
APP_BASE_URL = (os.getenv("APP_BASE_URL") or os.getenv("FRONTEND_BASE_URL") or "").strip()
if not APP_BASE_URL:
    APP_BASE_URL = "http://127.0.0.1:5173"

RESEND_ENDPOINT = "https://api.resend.com/emails"


def _require_resend_config() -> None:
    if not RESEND_API_KEY:
        raise RuntimeError("Missing RESEND_API_KEY")
    if not EMAIL_FROM:
        raise RuntimeError("Missing EMAIL_FROM")


def send_email_resend(
    to_email: str,
    subject: str,
    html: str,
    text: Optional[str] = None,
    reply_to: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Envía email por Resend.
    Lanza excepción si no hay config o si Resend responde error.
    """
    _require_resend_config()

    payload: Dict[str, Any] = {
        "from": EMAIL_FROM,
        "to": [to_email],
        "subject": subject,
        "html": html,
    }
    if text:
        payload["text"] = text
    if reply_to:
        payload["reply_to"] = reply_to
    if headers:
        payload["headers"] = headers

    r = requests.post(
        RESEND_ENDPOINT,
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=20,
    )

    # Resend suele contestar 200/201 cuando OK
    if r.status_code < 200 or r.status_code >= 300:
        raise RuntimeError(f"Resend error {r.status_code}: {r.text[:500]}")
    return r.json()


def build_verify_link(token: str) -> str:
    token = (token or "").strip()
    base = APP_BASE_URL.rstrip("/")
    return f"{base}/verify-email?token={token}"


def build_reset_link(token: str) -> str:
    token = (token or "").strip()
    base = APP_BASE_URL.rstrip("/")
    return f"{base}/reset-password?token={token}"


def send_verify_email(email: str, token: str) -> str:
    link = build_verify_link(token)
    subject = "Verifica tu correo — E-Vantis"

    html = f"""
    <div style="font-family:Arial,sans-serif;line-height:1.5">
      <h2>Verificación de correo</h2>
      <p>Para activar tu cuenta, verifica tu correo dando click aquí:</p>
      <p><a href="{link}" style="display:inline-block;padding:10px 14px;text-decoration:none;border-radius:10px;background:#1ECBE1;color:#0B132B;font-weight:700">Verificar correo</a></p>
      <p style="color:#666;font-size:12px">Si no fuiste tú, ignora este mensaje.</p>
      <p style="color:#666;font-size:12px">Link directo: {link}</p>
    </div>
    """.strip()

    send_email_resend(
        to_email=email,
        subject=subject,
        html=html,
        text=f"Verifica tu correo: {link}",
    )
    return link  # útil para QA si EVANTIS_RETURN_VERIFY_LINK=1


def send_reset_password_email(email: str, token: str) -> str:
    link = build_reset_link(token)
    subject = "Restablece tu contraseña — E-Vantis"

    html = f"""
    <div style="font-family:Arial,sans-serif;line-height:1.5">
      <h2>Restablecer contraseña</h2>
      <p>Recibimos una solicitud para restablecer tu contraseña.</p>
      <p><a href="{link}" style="display:inline-block;padding:10px 14px;text-decoration:none;border-radius:10px;background:#F4C95D;color:#0B132B;font-weight:800">Restablecer contraseña</a></p>
      <p style="color:#666;font-size:12px">Si no fuiste tú, ignora este mensaje.</p>
      <p style="color:#666;font-size:12px">Link directo: {link}</p>
    </div>
    """.strip()

    send_email_resend(
        to_email=email,
        subject=subject,
        html=html,
        text=f"Restablecer contraseña: {link}",
    )
    return link
