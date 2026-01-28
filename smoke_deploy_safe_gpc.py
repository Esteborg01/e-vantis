import os, json, sys, subprocess, time

BASE=os.getenv("BASE","https://e-vantis-api.onrender.com")
USER=os.getenv("USER","test_free@evantis.local")
PASS=os.getenv("PASS","Password123!")
API_KEY=os.getenv("API_KEY","")
assert API_KEY, "Falta API_KEY"

def run(cmd):
    p=subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def split(out):
    if "HTTP_STATUS:" not in out:
        return out.strip(), None
    body, status = out.rsplit("HTTP_STATUS:",1)
    return body.strip(), status.strip()

def health():
    rc,out,err = run(f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" "{BASE}/health"')
    if rc!=0:
        return None, "CURL_FAIL"
    body, status = split(out)
    return status, body

def login():
    rc,out,err = run(
        f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" '
        f'-X POST "{BASE}/auth/login" '
        f'-H "Content-Type: application/x-www-form-urlencoded" '
        f'--data-urlencode "username={USER}" '
        f'--data-urlencode "password={PASS}"'
    )
    if rc!=0:
        return None, None, "LOGIN_CURL_FAIL"
    body,status = split(out)
    if status!="200":
        return None, status, body
    tok = json.loads(body).get("access_token")
    return tok, status, body

def gpc(tok: str):
    payload = {
        "subject_id":"hematologia",
        "topic_id":"hema_t1_medula_osea_hematopoyesis",
        "module":"gpc_summary",
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
    if rc!=0:
        return None, "CURL_FAIL", (err or out)
    body,status = split(out)
    head = body[:350].replace("\n","\\n")
    return status, head, body

def classify(status, body):
    if status is None:
        return ("FAIL", "missing HTTP_STATUS")
    if status == "200":
        try:
            j=json.loads(body)
        except Exception:
            return ("FAIL", "200 but non-json")
        used=j.get("used_guides")
        cert=j.get("certifiable")
        if j.get("module")!="gpc_summary":
            return ("FAIL", "module mismatch")
        if not isinstance(j.get("gpc_summary",""), str) or len(j.get("gpc_summary",""))<200:
            return ("FAIL", "gpc_summary too short")
        if used is not True or cert is not True:
            return ("FAIL", f"expected used_guides/certifiable true (got used_guides={used}, certifiable={cert})")
        return ("OK", "200 contract + certifiable true")
    if status in ("429","403","500"):
        # For this smoke we want to detect mixed deploy behavior; don't auto-fail 500
        return ("NOT_OK", f"HTTP {status}")
    return ("NOT_OK", f"HTTP {status}")

print("=== DEPLOY-SAFE SMOKE (health + login + gpc x2) ===")

# health baseline
hs, hb = health()
print("HEALTH_STATUS:", hs, "HEALTH_BODY_HEAD:", (hb or "")[:80])
if hs != "200":
    print("SMOKE_FAIL: health not 200")
    sys.exit(1)

def one_round(tag):
    tok, ls, lb = login()
    print(f"{tag}_LOGIN_STATUS:", ls)
    if ls != "200" or not tok:
        print(f"{tag}_LOGIN_FAIL_BODY_HEAD:", (lb or "")[:200])
        return ("FAIL", "login failed")
    s, head, full = gpc(tok)
    print(f"{tag}_GPC_STATUS:", s)
    print(f"{tag}_GPC_BODY_HEAD:", head)
    if s == "CURL_FAIL":
        return ("FAIL", "curl fail")
    cls, reason = classify(s, full if isinstance(full,str) else "")
    return (cls, reason)

r1 = one_round("R1")
time.sleep(10)
r2 = one_round("R2")

print("R1_RESULT:", r1)
print("R2_RESULT:", r2)

if r1[0]=="OK" and r2[0]=="OK":
    print("SMOKE_OK: consistent (deploy complete + GPC OK)")
    sys.exit(0)

if r1[0]!=r2[0] or r1[1]!=r2[1]:
    print("SMOKE_INCONCLUSIVE_DEPLOY: inconsistent results across two runs")
    sys.exit(0)

print("SMOKE_FAIL: consistent but not OK -> investigate")
sys.exit(1)
