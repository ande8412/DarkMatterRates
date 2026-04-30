---
name: dmerates-onboarding
description: Use when chatting about how the DMeRates repository works, how to validate QCDark2/SRDM paths, or where to find the relevant architecture, data, and physics context.
---

# DMeRates Onboarding

Load only the context needed for the question:

1. Start with `AGENTS.md` for architecture, API conventions, and data layout.
2. Read `README.md` for user-facing setup and data-source notes.
3. Use `agents/README.md` and `agents/next_merge_run_order.md` for current runbook and merge-order context.
4. Read `tests/current_status.md` for support state and missing-scope status when that file exists.
5. Open QCDark2 validation notebooks under `tests/` only if physics-validation details are required.

Guardrails:

- Public API masses are MeV, including SRDM examples.
- SRDM internals/manifests use eV keys.
- QCDark2 calls must specify `screening='rpa'` or `screening='none'`.
