import os, json, sys, subprocess

BASE=os.getenv("BASE","https://e-vantis-api.onrender.com")
USER=os.getenv("USER","test_free@evantis.local")
PASS=os.getenv("PASS","Password123!")
API_KEY=os.getenv("API_KEY","")
assert API_KEY, "Falta API_KEY"

def run(cmd):
    p=subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()

def split(out):
    body,status = out.rsplit("HTTP_STATUS:",1)
    return body.strip(), status.strip()

# LOGIN
rc,out,err = run(
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" '
    f'-X POST "{BASE}/auth/login" '
    f'-H "Content-Type: application/x-www-form-urlencoded" '
    f'--data-urlencode "username={USER}" '
    f'--data-urlencode "password={PASS}"'
)
if rc!=0:
    print("LOGIN_CURL_FAIL"); print(err or out); sys.exit(1)

body,status = split(out)
print("LOGIN_STATUS:", status)
if status!="200":
    print("LOGIN_FAIL"); print(body); sys.exit(1)

tok = json.loads(body).get("access_token")
if not tok:
    print("LOGIN_MISSING_TOKEN"); sys.exit(1)

# EXAM_CLINICO (debe permitir en PRO)
payload = {
  "subject_id":"hematologia",
  "topic_id":"hema_t1_medula_osea_hematopoyesis",
  "module":"exam_clinico",
  "duration_minutes":20,
  "level":"auto",
  "style":"magistral",
  "exam_clinico_context": True,
  "num_questions": 8
}

rc,out,err = run(
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/teach/curriculum" '
    f'-H "Authorization: Bearer {tok}" '
    f'-H "X-API-Key: {API_KEY}" '
    f'-H "Content-Type: application/json" '
    f"-d '{json.dumps(payload)}'"
)
if rc!=0:
    print("EXAM_CLINICO_CURL_FAIL"); print(err or out); sys.exit(1)

body,status = split(out)
print("EXAM_CLINICO_STATUS:", status)
print("EXAM_CLINICO_BODY_HEAD:", body[:300].replace("\n","\\n"))

if status=="200":
    j=json.loads(body)
    assert j.get("module")=="exam_clinico", "module != exam_clinico"
    assert "exam_clinico" in j and isinstance(j["exam_clinico"], str) and len(j["exam_clinico"])>200, "exam_clinico missing/short"
    print("SMOKE_OK: EXAM_CLINICO allowed + contract valid (PRO)")
    sys.exit(0)

print("SMOKE_FAIL: expected 200 for PRO EXAM_CLINICO")
print(body)
sys.exit(1)
