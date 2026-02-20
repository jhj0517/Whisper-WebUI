# AGENTS.md

## Operating Model
This repository follows an evidence-first, zero-external-API-cost workflow.
Use GitHub Copilot coding agent and Codex app/IDE/CLI for implementation and review, then verify with deterministic commands before merge.

## Risk Policy
- Default merge policy: human-reviewed only.
- Use explicit risk labels: `risk:low`, `risk:medium`, `risk:high`.
- High-risk changes require rollback notes in the PR.

## Canonical Verification Command
Run this command before claiming completion:

```bash
bash scripts/verify
```

## Scope Guardrails
- Keep changes small and task-focused.
- Do not commit secrets or local runtime artifacts.
- Prefer tests/docs updates together with behavior changes.

## Agent Queue Contract
- Intake issues via `.github/ISSUE_TEMPLATE/agent_task.yml`.
- Queue work by adding `agent:ready` label.
- Queue workflow will post a task packet and notify `@copilot`.

## Queue Trigger Warning
Applying label `agent:ready` triggers the queue workflow immediately.
