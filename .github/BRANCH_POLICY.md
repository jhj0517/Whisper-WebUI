# Branch Naming and Default Branch Policy

## Canonical Default Branch
The canonical default branch name for this repository is `main`.

## Transition Rule
If the repository currently uses `master`, transition to `main` during Wave-2 rollout and keep workflow compatibility for one transition cycle before removing `master` triggers.

## Working Branches
- Governance bootstrap branch: `chore/governance-wave2-bootstrap`
- Hardening branch: `chore/governance-wave2-hardening`
- KPI/protection branch: `chore/governance-wave2-kpi-protection`

## Merge Rules
- Human-reviewed merges only.
- Squash merge by default.
- No direct pushes to default branch.
