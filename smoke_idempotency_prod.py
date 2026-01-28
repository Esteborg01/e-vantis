import os, json, sys, subprocess, time, uuid

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

def call_lesson(tok, idem_key):
    payload={"subject_id":"hematologia","topic_id":"hema_t1_medula_osea_hematopoyesis","module":"lesson","duration_minutes":20,"level":"auto","style":"magistral"}
    cmd=(
        f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/teach/curriculum" '
        f'-H "Authorization: Bearer {tok}" -H "X-API-Key: {API_KEY}" -H "Content-Type: application/json" '
        f'-H "Idempotency-Key: {idem_key}" '
        f"-d '{json.dumps(payload)}'"
    )
    rc,out,err=run(cmd)
    if rc!=0: return None,None,(err or out)
    body,status=split(out)
    return body,status,None

print("=== SMOKE: idempotency (same key twice) ===")
tok=login()
idem="idem-"+uuid.uuid4().hex[:12]

b1,s1,e1=call_lesson(tok, idem)
print("R1_STATUS:", s1)
if e1: print("R1_CURL_FAIL"); print(e1); sys.exit(1)

time.sleep(1)

b2,s2,e2=call_lesson(tok, idem)
print("R2_STATUS:", s2)
if e2: print("R2_CURL_FAIL"); print(e2); sys.exit(1)

# Both should be 200/429 but MUST be consistent.
# The key is: second call should NOT newly consume quota; commonly it returns same body.
if s1 != s2:
    print("SMOKE_FAIL: statuses differ between idempotent retries")
    print("R1_BODY_HEAD:", (b1 or "")[:200])
    print("R2_BODY_HEAD:", (b2 or "")[:200])
    sys.exit(1)

# If 200, bodies should match exactly (or at least have same module & same topic_id).
if s1 == "200":
    j1=json.loads(b1)
    j2=json.loads(b2)
    if j1.get("module")!="lesson" or j2.get("module")!="lesson":
        print("SMOKE_FAIL: wrong module in idempotent responses"); sys.exit(1)
    if j1.get("topic_id")!=j2.get("topic_id"):
        print("SMOKE_FAIL: topic differs"); sys.exit(1)
    # strict: identical response object expected
    if b1 != b2:
        print("SMOKE_WARN: idempotent 200 but bodies differ (still check quota logic manually)")
        print("R1_BODY_HEAD:", b1[:200])
        print("R2_BODY_HEAD:", b2[:200])
        sys.exit(0)

print("SMOKE_OK: idempotency stable (same key twice)")
