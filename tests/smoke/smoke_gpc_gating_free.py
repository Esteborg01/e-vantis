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

tok = json.loads(body).get("access_token")
if not tok:
    print("LOGIN_MISSING_TOKEN"); sys.exit(1)

# --- GPC SUMMARY (debe estar GATED en plan Free) ---
payload = {
    "subject_id": "hematologia",
    "topic_id": "hema_t1_medula_osea_hematopoyesis",
    "module": "gpc_summary",
    "duration_minutes": 20,
    "level": "auto",
    "style": "magistral"
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
    print("GPC_CURL_FAIL"); print(err or out); sys.exit(1)

body, status = split(out)
print("GPC_STATUS:", status)
print("GPC_BODY_HEAD:", body[:300].replace("\n", "\\n"))

# PASS esperado: 403 por gating
if status == "403":
    j = json.loads(body)
    detail = (j.get("detail") or "").lower() if isinstance(j, dict) else ""
    if ("gpc" in detail) and (("pro" in detail) or ("premium" in detail)):
        print("SMOKE_OK: GPC gated correctly for Free plan")
        sys.exit(0)

    print("SMOKE_FAIL: 403 but unexpected detail")
    print(body)
    sys.exit(1)

print("SMOKE_FAIL: GPC gating incorrect (expected 403)")
print(body)
sys.exit(1)
