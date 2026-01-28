import os, json, sys, subprocess

BASE = os.getenv("BASE", "https://e-vantis-api.onrender.com")
USER = os.getenv("USER", "test_free@evantis.local")
PASS = os.getenv("PASS", "Password123!")
API_KEY = os.getenv("API_KEY", "")

assert API_KEY, "Falta API_KEY: export API_KEY=ev_..."

def run(cmd: str):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()

def split_status(out: str):
    if "HTTP_STATUS:" not in out:
        return out.strip(), None
    body, status = out.rsplit("HTTP_STATUS:", 1)
    return body.strip(), status.strip()

# ----------------------------
# LOGIN
# ----------------------------
rc, out, err = run(
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" '
    f'-X POST "{BASE}/auth/login" '
    f'-H "Content-Type: application/x-www-form-urlencoded" '
    f'--data-urlencode "username={USER}" '
    f'--data-urlencode "password={PASS}"'
)
if rc != 0:
    print("LOGIN_CURL_FAIL"); print(err or out); sys.exit(1)

body, status = split_status(out)
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

# ----------------------------
# SUBJECT NOT FOUND
# ----------------------------
payload = {
    "subject_id": "_subject_inexistente_zzz",
    "topic_id": "hema_t1_medula_osea_hematopoyesis",
    "module": "lesson",
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
    print("CALL_CURL_FAIL"); print(err or out); sys.exit(1)

body, status = split_status(out)
print("CALL_STATUS:", status)
print("CALL_BODY_HEAD:", body[:300].replace("\n","\\n"))

# PASS esperado: 404 Subject not found
if status == "404":
    try:
        j = json.loads(body)
    except Exception:
        print("SMOKE_FAIL: 404 but non-json"); print(body); sys.exit(1)

    detail = (j.get("detail") or "").lower() if isinstance(j, dict) else ""
    if "subject not found" in detail:
        print("SMOKE_OK: subject not found returns 404")
        sys.exit(0)

    print("SMOKE_FAIL: 404 but unexpected detail")
    print(body)
    sys.exit(1)

print("SMOKE_FAIL: expected 404 for subject not found")
print(body)
sys.exit(1)
