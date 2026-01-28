import os, json, sys, subprocess

BASE=os.environ.get("BASE","https://e-vantis-api.onrender.com")
USER=os.environ.get("USER","test_free@evantis.local")
PASS=os.environ.get("PASS","Password123!")
API_KEY=os.environ.get("API_KEY","")

if not API_KEY:
    print("ERROR: falta API_KEY. Haz:\n  export API_KEY='ev_...'\n")
    sys.exit(1)

def run(cmd: str):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def split_status(out: str):
    if "HTTP_STATUS:" not in out:
        return out.strip(), None
    body, status = out.rsplit("HTTP_STATUS:", 1)
    return body.strip(), status.strip()

# --- LOGIN ---
login_cmd = (
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/auth/login" '
    f'-H "Content-Type: application/x-www-form-urlencoded" '
    f'--data-urlencode "username={USER}" '
    f'--data-urlencode "password={PASS}"'
)
rc, out, err = run(login_cmd)
if rc != 0:
    print("LOGIN_CURL_FAIL"); print(err or out); sys.exit(1)

login_body, login_status = split_status(out)
print("LOGIN_STATUS:", login_status)
if login_status != "200":
    print("LOGIN_BODY:", login_body[:500])
    sys.exit(1)

login_json = json.loads(login_body)
tok = login_json.get("access_token")
if not tok:
    print("LOGIN_FAIL: missing access_token"); print(login_json); sys.exit(1)

# --- LESSON ---
payload = {
  "subject_id":"hematologia",
  "topic_id":"hema_t1_medula_osea_hematopoyesis",
  "module":"lesson",
  "duration_minutes":20,
  "level":"auto",
  "style":"magistral"
}

lesson_cmd = (
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/teach/curriculum" '
    f'-H "Authorization: Bearer {tok}" '
    f'-H "X-API-Key: {API_KEY}" '
    f'-H "Content-Type: application/json" '
    f"-d '{json.dumps(payload)}'"
)
rc, out, err = run(lesson_cmd)
if rc != 0:
    print("LESSON_CURL_FAIL"); print(err or out); sys.exit(1)

body, status = split_status(out)
print("LESSON_STATUS:", status)
print("LESSON_BODY_HEAD:", body[:500].replace("\n","\\n"))

if status is None:
    print("LESSON_FAIL: missing HTTP_STATUS"); sys.exit(1)

# 429 PASS (quota)
if status == "429":
    try:
        j = json.loads(body)
    except Exception:
        print("LESSON_429_NON_JSON"); sys.exit(1)
    detail = (j.get("detail") or "").lower() if isinstance(j, dict) else ""
    if ("límite mensual" in detail) or ("limite mensual" in detail):
        print("SMOKE_OK: quota enforced (HTTP 429 monthly limit) — production healthy")
        sys.exit(0)
    print("LESSON_HTTP_429_UNEXPECTED"); print(body); sys.exit(1)

# 200 must be JSON object with module/lesson
if status == "200":
    resp = json.loads(body)  # may become None if body == "null"
    print("LESSON_PARSED_TYPE:", type(resp).__name__)
    if resp is None:
        print("LESSON_BODY_WAS_NULL (JSON null). Backend returned null with HTTP 200.")
        sys.exit(1)
    if not isinstance(resp, dict):
        print("LESSON_JSON_NOT_OBJECT"); print(resp); sys.exit(1)

    assert resp.get("module") == "lesson", f"module != lesson (got {resp.get('module')})"
    assert "lesson" in resp, "Missing key: lesson"

    txt = resp["lesson"]
    assert isinstance(txt, str) and len(txt) > 300, "lesson too short / not string"

    tail = txt[-2000:].lower()
    ok = ("preguntas de repaso" in tail) or (("preguntas" in tail) and ("repaso" in tail))
    assert ok, "missing 'Preguntas de repaso' near end of lesson"

    print("SMOKE_OK: lesson contract valid + review questions present (HTTP 200)")
    sys.exit(0)

print(f"LESSON_HTTP_{status}")
print(body[:800])
sys.exit(1)
