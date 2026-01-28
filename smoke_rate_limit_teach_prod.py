import os, json, sys, subprocess, time

BASE = os.getenv("BASE", "https://e-vantis-api.onrender.com")
USER = os.getenv("USER", "test_free@evantis.local")
PASS = os.getenv("PASS", "Password123!")
API_KEY = os.getenv("API_KEY", "")

assert API_KEY, "Falta API_KEY"

def run(cmd: str):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def split(out: str):
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

body, status = split(out)
print("LOGIN_STATUS:", status)
if status != "200":
    print("LOGIN_FAIL"); print(body); sys.exit(1)

tok = json.loads(body).get("access_token")
if not tok:
    print("LOGIN_MISSING_TOKEN"); sys.exit(1)

payload = {
    "subject_id": "hematologia",
    "topic_id": "hema_t1_medula_osea_hematopoyesis",
    "module": "lesson",
    "duration_minutes": 20,
    "level": "auto",
    "style": "magistral",
}

def call_once(i: int):
    cmd = (
        f'curl -sS -w "\\nHTTP_STATUS:%{{http_code}}\\n" -X POST "{BASE}/teach/curriculum" '
        f'-H "Authorization: Bearer {tok}" '
        f'-H "X-API-Key: {API_KEY}" '
        f'-H "Content-Type: application/json" '
        f"-d '{json.dumps(payload)}'"
    )
    rc, out, err = run(cmd)
    if rc != 0:
        return None, f"CURL_FAIL {err or out}"
    b, s = split(out)
    head = b[:160].replace("\n", "\\n")
    return (s, head)

# Burst: 40 requests rápidas (límite declarado: 30/min)
rate_limit_hits = 0
monthly_quota_hits = 0
other_429_hits = 0
statuses = {}

for i in range(1, 41):
    s, head = call_once(i)
    if s is None:
        print("REQ_FAIL:", head)
        sys.exit(1)

    statuses[s] = statuses.get(s, 0) + 1

    if s == "429":
        # distinguimos cuota mensual vs rate limit por el texto
        try:
            j = json.loads(head.replace("\\n","\n"))  # head es truncated; puede no parsear
            detail = (j.get("detail") or "").lower()
        except Exception:
            detail = head.lower()

        if ("límite mensual" in detail) or ("limite mensual" in detail):
            monthly_quota_hits += 1
        elif ("rate" in detail) or ("too many" in detail) or ("por minuto" in detail) or ("limit" in detail):
            rate_limit_hits += 1
        else:
            other_429_hits += 1

    # micro-delay para no ser *demasiado* agresivo con red
    time.sleep(0.05)

print("STATUS_COUNTS:", statuses)
print("429_BREAKDOWN:", {"rate_limit": rate_limit_hits, "monthly_quota": monthly_quota_hits, "other": other_429_hits})

# PASS si encontramos rate limit real
if rate_limit_hits > 0:
    print("SMOKE_OK: rate limit enforced on /teach/curriculum")
    sys.exit(0)

# Si solo hay cuota mensual, no probamos rate limit
if monthly_quota_hits > 0 and (rate_limit_hits == 0):
    print("SMOKE_INCONCLUSIVE: only monthly quota triggered; rate limit not proven")
    sys.exit(0)

print("SMOKE_FAIL: no evidence of rate limiting (and no monthly quota either)")
sys.exit(1)
