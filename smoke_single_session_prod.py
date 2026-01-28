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

def login(label: str):
    cmd = (
        f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/auth/login" '
        f'-H "Content-Type: application/x-www-form-urlencoded" '
        f'--data-urlencode "username={USER}" '
        f'--data-urlencode "password={PASS}"'
    )
    rc, out, err = run(cmd)
    if rc != 0:
        print(f"{label}_LOGIN_CURL_FAIL"); print(err or out); sys.exit(1)

    body, status = split_status(out)
    print(f"{label}_LOGIN_STATUS:", status)

    if status != "200":
        print(f"{label}_LOGIN_HTTP_{status}"); print(body); sys.exit(1)

    j = json.loads(body)
    tok = j.get("access_token")
    if not tok:
        print(f"{label}_LOGIN_FAIL: missing access_token"); print(j); sys.exit(1)
    return tok

def call_lesson(tok: str):
    payload = {
      "subject_id":"hematologia",
      "topic_id":"hema_t1_medula_osea_hematopoyesis",
      "module":"lesson",
      "duration_minutes":20,
      "level":"auto",
      "style":"magistral"
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
        return None, None, (err or out)
    body, status = split_status(out)
    return body, status, None

# 1) Login #1 -> token A
tokA = login("A")

# 2) Validar que token A sirve (lesson debe dar 200 o 429)
bodyA, statusA, errA = call_lesson(tokA)
print("A_LESSON_STATUS:", statusA)
if errA:
    print("A_LESSON_CURL_FAIL"); print(errA); sys.exit(1)
if statusA not in ("200","429"):
    print("A_LESSON_UNEXPECTED"); print(bodyA[:800]); sys.exit(1)

# 3) Login #2 -> token B (esto debe revocar token A si single-session está activo)
tokB = login("B")

# 4) Token B debe servir (200 o 429)
bodyB, statusB, errB = call_lesson(tokB)
print("B_LESSON_STATUS:", statusB)
if errB:
    print("B_LESSON_CURL_FAIL"); print(errB); sys.exit(1)
if statusB not in ("200","429"):
    print("B_LESSON_UNEXPECTED"); print(bodyB[:800]); sys.exit(1)

# 5) Token A ahora DEBE FALLAR (401) si single-session enforced
bodyA2, statusA2, errA2 = call_lesson(tokA)
print("A2_LESSON_STATUS:", statusA2)
if errA2:
    print("A2_LESSON_CURL_FAIL"); print(errA2); sys.exit(1)

if statusA2 == "401":
    print("SMOKE_OK: single-session enforced (old token revoked)")
    sys.exit(0)

# Si no es 401, marcamos falla con info
print("SMOKE_FAIL: old token still valid; expected 401 after re-login")
print("A2_BODY_HEAD:", (bodyA2 or "")[:800])
sys.exit(1)
