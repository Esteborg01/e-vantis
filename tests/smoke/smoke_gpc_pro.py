import os, json, sys, subprocess

BASE=os.getenv("BASE","https://e-vantis-api.onrender.com")
USER=os.getenv("USER","test_free@evantis.local")
PASS=os.getenv("PASS","Password123!")
API_KEY=os.getenv("API_KEY","")

assert API_KEY, "Falta API_KEY: export API_KEY=ev_..."

def run(cmd: str):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()

def split_status(out: str):
    if "HTTP_STATUS:" not in out:
        return out.strip(), None
    body, status = out.rsplit("HTTP_STATUS:", 1)
    return body.strip(), status.strip()

# --- LOGIN ---
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

tok = json.loads(body).get("access_token")
if not tok:
    print("LOGIN_MISSING_TOKEN"); sys.exit(1)

# --- GPC SUMMARY (PRO: debe permitir 200) ---
payload = {
  "subject_id":"hematologia",
  "topic_id":"hema_t1_medula_osea_hematopoyesis",
  "module":"gpc_summary",
  "duration_minutes":20,
  "level":"auto",
  "style":"magistral"
}

rc, out, err = run(
    f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/teach/curriculum" '
    f'-H "Authorization: Bearer {tok}" '
    f'-H "X-API-Key: {API_KEY}" '
    f'-H "Content-Type: application/json" '
    f"-d '{json.dumps(payload)}'"
)
if rc != 0:
    print("GPC_CURL_FAIL"); print(err or out); sys.exit(1)

body, status = split_status(out)
print("GPC_STATUS:", status)
print("GPC_BODY_HEAD:", body[:300].replace("\n","\\n"))

# PASS A: 200 contract
if status == "200":
    try:
        resp = json.loads(body)
    except Exception:
        print("GPC_JSON_PARSE_FAIL"); print(body); sys.exit(1)

    assert isinstance(resp, dict), "GPC_JSON_NOT_OBJECT"
    assert resp.get("module") == "gpc_summary", f"module != gpc_summary (got {resp.get('module')})"
    assert "gpc_summary" in resp, "Missing key: gpc_summary"
    txt = resp["gpc_summary"]
    assert isinstance(txt, str) and len(txt) > 200, "gpc_summary too short / not string"

    # En gpc_summary debe usar web_search para ser "certifiable"
    assert resp.get("used_guides") in (True, False), "used_guides missing/not bool-ish"
    assert resp.get("certifiable") in (True, False), "certifiable missing/not bool-ish"
    if resp.get("certifiable") is not True:
        print("SMOKE_FAIL: gpc_summary should be certifiable=True when web_search used")
        print("used_guides:", resp.get("used_guides"), "certifiable:", resp.get("certifiable"))
        sys.exit(1)

    print("SMOKE_OK: GPC allowed + contract valid (PRO) + certifiable true")
    sys.exit(0)

# PASS B: 429 monthly limit enforced (prod healthy)
if status == "429":
    try:
        j = json.loads(body)
    except Exception:
        print("GPC_429_NON_JSON"); print(body); sys.exit(1)

    detail = (j.get("detail") or "").lower() if isinstance(j, dict) else ""
    if ("límite mensual" in detail) or ("limite mensual" in detail):
        print("SMOKE_OK: quota enforced (HTTP 429 monthly limit) — production healthy")
        sys.exit(0)

    print("GPC_HTTP_429_UNEXPECTED"); print(body); sys.exit(1)

print("SMOKE_FAIL: expected 200 (or 429 quota) for PRO gpc_summary")
print(body)
sys.exit(1)
