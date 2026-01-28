import os, json, sys, subprocess, uuid

BASE=os.getenv("BASE","https://e-vantis-api.onrender.com")
USER=os.getenv("USER","test_free@evantis.local")
PASS=os.getenv("PASS","Password123!")
API_KEY=os.getenv("API_KEY","")
assert API_KEY, "Falta API_KEY"

def run(cmd):
    p=subprocess.run(cmd,shell=True,capture_output=True,text=True)
    return p.returncode,p.stdout,p.stderr

def split(out):
    if "HTTP_STATUS:" not in out:
        return out.strip(), None
    b,s=out.rsplit("HTTP_STATUS:",1)
    return b.strip(), s.strip()

def login():
    rc,out,err=run(
        f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/auth/login" '
        f'-H "Content-Type: application/x-www-form-urlencoded" '
        f'--data-urlencode "username={USER}" --data-urlencode "password={PASS}"'
    )
    if rc!=0: print("LOGIN_CURL_FAIL"); print(err or out); sys.exit(1)
    body,status=split(out)
    print("LOGIN_STATUS:", status)
    if status!="200": print("LOGIN_FAIL"); print(body); sys.exit(1)
    tok=json.loads(body).get("access_token")
    if not tok: print("LOGIN_NO_TOKEN"); sys.exit(1)
    return tok

def revoke(tok):
    # Ajusta endpoint si tu backend usa otro (ej: /auth/logout)
    rc,out,err=run(
        f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/auth/revoke" '
        f'-H "Authorization: Bearer {tok}"'
    )
    if rc!=0: print("REVOKE_CURL_FAIL"); print(err or out); sys.exit(1)
    body,status=split(out)
    print("REVOKE_STATUS:", status)
    # muchos backends responden 200 o 204
    if status not in ("200","204"):
        print("REVOKE_UNEXPECTED"); print(body); sys.exit(1)

def call_lesson(tok):
    payload={"subject_id":"hematologia","topic_id":"hema_t1_medula_osea_hematopoyesis","module":"lesson","duration_minutes":20,"level":"auto","style":"magistral"}
    idem="idem-"+uuid.uuid4().hex[:12]
    rc,out,err=run(
        f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/teach/curriculum" '
        f'-H "Authorization: Bearer {tok}" -H "X-API-Key: {API_KEY}" -H "Content-Type: application/json" '
        f'-H "Idempotency-Key: {idem}" '
        f"-d '{json.dumps(payload)}'"
    )
    if rc!=0: print("LESSON_CURL_FAIL"); print(err or out); sys.exit(1)
    body,status=split(out)
    print("LESSON_STATUS:", status)
    print("LESSON_BODY_HEAD:", (body or "")[:200].replace("\n","\\n"))
    return body,status

print("=== SMOKE: revoke makes token invalid ===")
tok=login()

# 1) token debe servir (200/429) ANTES de revoke
_, st1 = call_lesson(tok)
if st1 not in ("200","429"):
    print("SMOKE_FAIL: token not usable before revoke"); sys.exit(1)

# 2) revoke
revoke(tok)

# 3) token DEBE fallar 401 después
body2, st2 = call_lesson(tok)
if st2 == "401":
    print("SMOKE_OK: revoke enforced (token invalid after revoke)")
    sys.exit(0)

print("SMOKE_FAIL: expected 401 after revoke, got", st2)
print("BODY:", body2[:800])
sys.exit(1)
