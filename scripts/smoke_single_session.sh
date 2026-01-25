#!/bin/bash
set -e
set +H 2>/dev/null || true

: "${BASE:=http://127.0.0.1:8000}"
: "${EMAIL:?Missing EMAIL}"
: "${PASS:?Missing PASS}"

echo "=== SMOKE: SINGLE SESSION + LOGOUT ==="

echo "1) Login A"
TOKEN_A=$(curl -s -X POST "$BASE/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=$EMAIL" \
  --data-urlencode "password=$PASS" \
| python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s "$BASE/auth/me" -H "Authorization: Bearer $TOKEN_A" >/dev/null
echo "OK TOKEN_A válido"

echo "2) Login B (revoca A)"
TOKEN_B=$(curl -s -X POST "$BASE/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=$EMAIL" \
  --data-urlencode "password=$PASS" \
| python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s "$BASE/auth/me" -H "Authorization: Bearer $TOKEN_B" >/dev/null
echo "OK TOKEN_B válido"

echo "3) TOKEN_A debe fallar"
OUT_A=$(curl -s "$BASE/auth/me" -H "Authorization: Bearer $TOKEN_A" || true)
echo "$OUT_A" | grep -qi "revocada"
echo "OK TOKEN_A revocado"

echo "4) Logout B"
curl -s -X POST "$BASE/auth/logout" -H "Authorization: Bearer $TOKEN_B" >/dev/null
echo "OK logout"

echo "5) TOKEN_B debe fallar"
OUT_B=$(curl -s "$BASE/auth/me" -H "Authorization: Bearer $TOKEN_B" || true)
echo "$OUT_B" | grep -qi "revocada"
echo "OK TOKEN_B revocado"

echo "=== TODO VERDE (SINGLE SESSION + LOGOUT) ==="
