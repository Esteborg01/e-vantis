#!/usr/bin/env bash
set -euo pipefail

BASE="https://e-vantis-api.onrender.com"

USER="test_free@evantis.local"
PASS='Password123!'
API_KEY="ev_1UhM1H8PwQAxZ2UaM3-21AAPSU47DPiQ"

SUBJECT_ID="hematologia"
TOPIC_ID="hema_t1_medula_osea_hematopoyesis"

STYLE="magistral"
LEVEL="auto"
DURATION=20

echo "==> Health"
curl -s "$BASE/health" | python -m json.tool >/dev/null
echo "OK: /health"

echo "==> Login"
TOKEN=$(
  curl -s -X POST "$BASE/auth/login" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    --data-urlencode "username=$USER" \
    --data-urlencode "password=$PASS" \
  | python -c 'import sys,json; print(json.load(sys.stdin)["access_token"])'
)

if [[ -z "${TOKEN:-}" ]]; then
  echo "ERROR: no se obtuvo TOKEN"
  exit 1
fi
echo "OK: token emitido"

echo "==> Lesson (teach/curriculum)"
LESSON_JSON=$(
  curl -s -X POST "$BASE/teach/curriculum" \
    -H "Authorization: Bearer $TOKEN" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
      \"subject_id\":\"$SUBJECT_ID\",
      \"topic_id\":\"$TOPIC_ID\",
      \"module\":\"lesson\",
      \"duration_minutes\":$DURATION,
      \"level\":\"$LEVEL\",
      \"style\":\"$STYLE\"
    }"
)

echo "$LESSON_JSON" | python -m json.tool >/dev/null

python - <<'PY' <<<"$LESSON_JSON"
import json,sys
obj=json.loads(sys.stdin.read())
assert obj.get("module")=="lesson", f"module inesperado: {obj.get('module')}"
assert "lesson" in obj and isinstance(obj["lesson"], str) and len(obj["lesson"])>200, "lesson vacío/corto"
assert obj.get("subject_id")=="hematologia", "subject_id inesperado"
assert obj.get("topic_id")=="hema_t1_medula_osea_hematopoyesis", "topic_id inesperado"
print("OK: lesson válido")
PY

echo "==> Exam (teach/curriculum)"
EXAM_JSON=$(
  curl -s -X POST "$BASE/teach/curriculum" \
    -H "Authorization: Bearer $TOKEN" \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
      \"subject_id\":\"$SUBJECT_ID\",
      \"topic_id\":\"$TOPIC_ID\",
      \"module\":\"exam\",
      \"duration_minutes\":$DURATION,
      \"level\":\"$LEVEL\",
      \"style\":\"$STYLE\"
    }"
)

echo "$EXAM_JSON" | python -m json.tool >/dev/null

python - <<'PY' <<<"$EXAM_JSON"
import json,sys
obj=json.loads(sys.stdin.read())
assert obj.get("module")=="exam", f"module inesperado: {obj.get('module')}"
assert "exam" in obj and isinstance(obj["exam"], str) and len(obj["exam"])>200, "exam vacío/corto o clave distinta"
print("OK: exam válido")
PY

echo "==> SMOKE OK (lesson + exam)"
