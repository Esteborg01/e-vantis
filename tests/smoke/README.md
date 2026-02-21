# Smoke tests — E-VANTIS Backend

Estos scripts validan el comportamiento del backend en producción o staging.

Cobertura:
- autenticación, single-session, revoke, logout
- lesson, exam, exam_clinico (gating por plan)
- gpc_summary (deploy-safe)
- cuotas mensuales, rate limit
- api key inválida, subject inexistente, idempotency

Uso:
Ejecución manual como parte del playbook operativo.

