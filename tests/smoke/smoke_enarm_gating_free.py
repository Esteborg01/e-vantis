import os, json, sys, subprocess

BASE = os.getenv("BASE", "https://e-vantis-api.onrender.com")
USER = os.getenv("USER", "test_free@evantis.local")
PASS = os.getenv("PASS", "Password123!")
API_KEY = os.getenv("API_KEY", "")

assert API_KEY, "Falta API_KEY"

def run(cmd: str):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def split(out: str):
    if "HTTP_STATUS:" not in out:
        return out.strip(), None
    body, status = out.rsplit("HTTP_STATUS:", 1)
    return body.strip(), status.strip()

# --- LOGIN ---
rc, out, err = run(
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" '
    f'-X POST "{BASE}/auth/login" '
    f'-H "Content-Type: application/x-www-form-urlencoded" '
    f'--data-urlencode "username={USER}" '
    f'--data-urlencode "password={PASS}"'
)
if rc != 0:
    print("LOGIN_CURL_FAIL"); print(err or out); sys.exit(1)

body, status = split(out)
print("LOGIN_STATUS:", status)
if status != "200":
    print("LOGIN_FAIL"); print(body); sys.exit(1)

try:
    j = json.loads(body)
except Exception:
    print("LOGIN_JSON_PARSE_FAIL"); print(body); sys.exit(1)

tok = j.get("access_token")
if not tok:
    print("LOGIN_MISSING_TOKEN"); print(j); sys.exit(1)

# --- EXAM_CLINICO (debe estar GATED en plan Free) ---
payload = {
    "subject_id": "hematologia",
    "topic_id": "hema_t1_medula_osea_hematopoyesis",
    "module": "exam_clinico",
    "duration_minutes": 20,
    "level": "auto",
    "style": "magistral",
    "exam_clinico_context": True
}

cmd = (
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/teach/curriculum" '
    f'-H "Authorization: Bearer {tok}" '
    f'-H "X-API-Key: {API_KEY}" '
    f'-H "Content-Type: application/json" '
    f"-d '{json.dumps(payload)}'"
)

rc, out, err = run(cmd)
if rc != 0:
    print("EXAM_CLINICO_CURL_FAIL"); print(err or out); sys.exit(1)

body, status = split(out)
print("EXAM_CLINICO_STATUS:", status)
print("EXAM_CLINICO_BODY_HEAD:", body[:300].replace("\n", "\\n"))

# PASS esperado: 403 por gating
if status == "403":
    try:
        jj = json.loads(body)
    except Exception:
        print("EXAM_CLINICO_403_NON_JSON"); print(body); sys.exit(1)

    detail = (jj.get("detail") or "").lower() if isinstance(jj, dict) else ""
    if ("exam_clinico" in detail) and (("pro" in detail) or ("premium" in detail)):
        print("SMOKE_OK: EXAM_CLINICO gated correctly for Free plan")
        sys.exit(0)

    print("SMOKE_FAIL: 403 but unexpected detail")
    print(body)
    sys.exit(1)

print("SMOKE_FAIL: EXAM_CLINICO gating incorrect (expected 403)")
print(body)
sys.exit(1)
