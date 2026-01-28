import os, json, sys, subprocess, time

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

def health():
    cmd = f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" "{BASE}/health"'
    rc, out, err = run(cmd)
    if rc != 0:
        print("HEALTH_CURL_FAIL"); print(err or out); sys.exit(1)
    body, status = split_status(out)
    print("HEALTH_STATUS:", status)
    if status != "200":
        print("HEALTH_BAD"); print(body); sys.exit(1)
    print("HEALTH_BODY:", body[:200])

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
        print("LOGIN_FAIL"); print(body); sys.exit(1)
    j = json.loads(body)
    tok = j.get("access_token")
    if not tok:
        print("LOGIN_MISSING_TOKEN"); print(j); sys.exit(1)
    return tok

def lesson(tok: str):
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
        print("LESSON_CURL_FAIL"); print(err or out); sys.exit(1)
    body, status = split_status(out)
    print("LESSON_STATUS:", status)
    print("LESSON_BODY_HEAD:", body[:200].replace("\n","\\n"))
    if status not in ("200","429"):
        print("LESSON_UNEXPECTED"); print(body[:800]); sys.exit(1)

print("=== SMOKE: health/login/lesson x2 ===")
health()
tok1 = login()
lesson(tok1)

print("=== sleep 3s and repeat ===")
time.sleep(3)

health()
tok2 = login()
lesson(tok2)

print("SMOKE_OK: stable across two runs (use this before/after redeploy)")
