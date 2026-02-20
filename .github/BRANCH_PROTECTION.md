# Branch Protection Policy

## Default Branch Requirements
The default branch must enforce:
- Pull request required before merge
- At least 1 human approval
- Required status check: `verify`
- Branch must be up to date before merge
- No force pushes
- No deletions

## Risk Routing
- `risk:low`: standard gate
- `risk:medium`: standard gate + explicit rollback section
- `risk:high`: standard gate + additional human scrutiny and explicit rollback plan

## Review Policy
- Human-reviewed merges only
- Agent-generated PRs are never auto-merged

## Escaped Regression Tracking
Mark post-merge regressions with label `escaped-regression` and include link to causative PR.
