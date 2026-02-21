import os, json, sys, subprocess

BASE=os.getenv("BASE","https://e-vantis-api.onrender.com")
USER=os.getenv("USER","test_free@evantis.local")
PASS=os.getenv("PASS","Password123!")
API_KEY=os.getenv("API_KEY","")

assert API_KEY, "Falta API_KEY: export API_KEY=ev_..."

def run(cmd: str):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def split(out: str):
    if "HTTP_STATUS:" not in out:
        return out.strip(), None
    body, status = out.rsplit("HTTP_STATUS:", 1)
    return body.strip(), status.strip()

# --- LOGIN ---
rc,out,err = run(
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" '
    f'-X POST "{BASE}/auth/login" '
    f'-H "Content-Type: application/x-www-form-urlencoded" '
    f'--data-urlencode "username={USER}" '
    f'--data-urlencode "password={PASS}"'
)
if rc != 0:
    print("LOGIN_CURL_FAIL"); print(err or out); sys.exit(1)

body,status = split(out)
print("LOGIN_STATUS:", status)
if status != "200":
    print("LOGIN_FAIL"); print(body); sys.exit(1)

tok = json.loads(body).get("access_token")
if not tok:
    print("LOGIN_MISSING_TOKEN"); sys.exit(1)

# --- UNSUPPORTED MODULE ---
payload = {
  "subject_id":"hematologia",
  "topic_id":"hema_t1_medula_osea_hematopoyesis",
  "module":"invalid_module_xyz",
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

rc,out,err = run(cmd)
if rc != 0:
    print("CALL_CURL_FAIL"); print(err or out); sys.exit(1)

body,status = split(out)
print("CALL_STATUS:", status)
print("CALL_BODY_HEAD:", body[:300].replace("\n","\\n"))

# PASS esperado: 422
if status == "422":
    try:
        j = json.loads(body)
    except Exception:
        print("SMOKE_FAIL: 422 but non-json body"); print(body); sys.exit(1)

    detail = j.get("detail")

    # Caso A: tu backend devuelve string
    if isinstance(detail, str):
        if "módulo no soportado" in detail.lower() or "modulo no soportado" in detail.lower():
            print("SMOKE_OK: unsupported module rejected (422) — backend validation")
            sys.exit(0)

    # Caso B: Pydantic devuelve lista de errores
    if isinstance(detail, list):
        s = json.dumps(detail, ensure_ascii=False).lower()
        if ("input should be" in s) and ("lesson" in s) and ("exam" in s) and ("exam_clinico" in s) and ("gpc_summary" in s):
            print("SMOKE_OK: unsupported module rejected (422) — pydantic literal validation")
            sys.exit(0)

    print("SMOKE_FAIL: 422 but unexpected detail shape/content")
    print(body)
    sys.exit(1)

