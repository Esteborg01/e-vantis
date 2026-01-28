import os, json, sys, subprocess

BASE = os.getenv("BASE", "https://e-vantis-api.onrender.com")
USER = os.getenv("USER", "test_free@evantis.local")
PASS = os.getenv("PASS", "Password123!")
API_KEY = os.getenv("API_KEY", "")

assert API_KEY, "Falta API_KEY: export API_KEY=ev_..."

def run(cmd: str):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def split_status(out: str):
    if "HTTP_STATUS:" not in out:
        return out.strip(), None
    body, status = out.rsplit("HTTP_STATUS:", 1)
    return body.strip(), status.strip()

def login():
    cmd = (
        f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/auth/login" '
        f'-H "Content-Type: application/x-www-form-urlencoded" '
        f'--data-urlencode "username={USER}" '
        f'--data-urlencode "password={PASS}"'
    )
    rc, out, err = run(cmd)
    if rc != 0:
        print("LOGIN_CURL_FAIL"); print(err or out); sys.exit(1)

    body, status = split_status(out)
    print("LOGIN_STATUS:", status)

    if status != "200":
        print(f"LOGIN_HTTP_{status}"); print(body); sys.exit(1)

    try:
        j = json.loads(body)
    except Exception:
        print("LOGIN_JSON_PARSE_FAIL"); print(body); sys.exit(1)

    tok = j.get("access_token")
    if not tok:
        print("LOGIN_FAIL: missing access_token"); print(j); sys.exit(1)
    return tok

tok = login()

payload = {
  "subject_id":"hematologia",
  "topic_id":"hema_t1_medula_osea_hematopoyesis",
  "module":"exam",
  "duration_minutes":20,
  "level":"auto",
  "style":"magistral"
}

exam_cmd = (
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/teach/curriculum" '
    f'-H "Authorization: Bearer {tok}" '
    f'-H "X-API-Key: {API_KEY}" '
    f'-H "Content-Type: application/json" '
    f"-d '{json.dumps(payload)}'"
)

rc, out, err = run(exam_cmd)
if rc != 0:
    print("EXAM_CURL_FAIL"); print(err or out); sys.exit(1)

body, status = split_status(out)
print("EXAM_STATUS:", status)
print("EXAM_BODY_HEAD:", body[:500].replace("\n","\\n"))

if status is None:
    print("EXAM_FAIL: missing HTTP_STATUS trailer"); sys.exit(1)

# PASS A: 200 contract
if status == "200":
    try:
        resp = json.loads(body)
    except Exception:
        print("EXAM_JSON_PARSE_FAIL"); print(body); sys.exit(1)

    if not isinstance(resp, dict):
        print("EXAM_JSON_NOT_OBJECT"); print(resp); sys.exit(1)

    assert resp.get("module") == "exam", f"module != exam (got {resp.get('module')})"
    assert "exam" in resp, "Missing key: exam"
    assert isinstance(resp["exam"], str) and len(resp["exam"]) > 50, "exam too short / not string"

    # Guardrail: exam should not come in lesson
    assert ("lesson" not in resp) or (resp.get("lesson") in ("", None)), "lesson should not carry exam payload"

    print("SMOKE_OK: exam contract valid (HTTP 200)")
    sys.exit(0)

# PASS B: 429 monthly limit enforced (prod healthy)
if status == "429":
    try:
        j = json.loads(body)
    except Exception:
        print("EXAM_429_NON_JSON"); print(body); sys.exit(1)

    detail = (j.get("detail") or "").lower() if isinstance(j, dict) else ""
    if ("límite mensual" in detail) or ("limite mensual" in detail):
        print("SMOKE_OK: quota enforced (HTTP 429 monthly limit) — production healthy")
        sys.exit(0)

    print("EXAM_HTTP_429_UNEXPECTED"); print(body); sys.exit(1)

print(f"EXAM_HTTP_{status}")
print(body)
sys.exit(1)
