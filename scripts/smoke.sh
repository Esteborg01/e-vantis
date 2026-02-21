#!/bin/bash
set -e
set +H 2>/dev/null || true

: "${BASE:=http://127.0.0.1:8000}"
: "${EMAIL:?Missing EMAIL}"
: "${PASS:?Missing PASS}"
: "${API_KEY:?Missing API_KEY}"

echo "=== SMOKE: E-VANTIS (sesion + teach full) ==="

echo "1) Sesión única"
./scripts/smoke_single_session.sh

echo "2) Topic_id (desde curriculum local)"
TOPIC_ID=$(python -c "import json; d=json.load(open('curriculum/evantis.curriculum.v1.json','r',encoding='utf-8')); s=next(x for x in d['subjects'] if x['id']=='urgencias'); print(s['blocks'][0]['macro_topics'][0]['id'])")
echo "TOPIC_ID=$TOPIC_ID"

echo "3) Login (token para teach)"
TOKEN=$(curl -s -X POST "$BASE/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=$EMAIL" \
  --data-urlencode "password=$PASS" \
| python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "4) auth/me (plan real)"
PLAN=$(curl -s "$BASE/auth/me" -H "Authorization: Bearer $TOKEN" | python -c "import sys,json; print(json.load(sys.stdin)['plan'])")
echo "PLAN=$PLAN"

post_teach () {
  local module="$1"
  local extra="$2"
  curl -s -X POST "$BASE/teach/curriculum" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"subject_id\":\"urgencias\",\"topic_id\":\"$TOPIC_ID\",\"module\":\"$module\"${extra}}" \
    | python -m json.tool > /dev/null
}

echo "5) lesson (200 esperado)"
post_teach "lesson" ""
echo "OK lesson"

echo "6) exam (200 esperado)"
post_teach "exam" ""
echo "OK exam"

if [ "$PLAN" = "pro" ] || [ "$PLAN" = "premium" ]; then
  echo "7) exam_clinico (200 esperado en pro/premium)"
  post_teach "exam_clinico" ",\"exam_clinico_context\":true,\"num_questions\":5"
  echo "OK exam_clinico"

  echo "8) gpc_summary (200 esperado en pro/premium)"
  post_teach "gpc_summary" ",\"use_guides\":false"
  echo "OK gpc_summary"
else
  echo "7) exam_clinico (403 esperado en free)"
  set +e
  RES=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/teach/curriculum" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"subject_id\":\"urgencias\",\"topic_id\":\"$TOPIC_ID\",\"module\":\"exam_clinico\",\"exam_clinico_context\":true,\"num_questions\":5}")
  set -e
  [ "$RES" = "403" ] && echo "OK exam_clinico 403" || (echo "FAIL exam_clinico expected 403 got $RES" && exit 1)

  echo "8) gpc_summary (403 esperado en free)"
  set +e
  RES=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/teach/curriculum" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"subject_id\":\"urgencias\",\"topic_id\":\"$TOPIC_ID\",\"module\":\"gpc_summary\",\"use_guides\":true}")
  set -e
  [ "$RES" = "403" ] && echo "OK gpc_summary 403" || (echo "FAIL gpc_summary expected 403 got $RES" && exit 1)
fi

echo "=== TODO VERDE (sesion + teach full) ==="
