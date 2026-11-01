# Engineering guidance

Apply `$software-design-patterns` when designing, implementing, extending, refactoring, or reviewing components. Read the relevant decisions, architecture, and design contracts. Verify the open–closed principle for every affected component before declaring the build complete.

If unavailable in discovery, read `${CODEX_HOME:-$HOME/.codex}/skills/software-design-patterns/SKILL.md` and report a missing dependency accurately.

Read `PRD.md`, `BUILD_BRIEF.md`, `architecture.md`, `design.md`, `decisions.md`, and the current `PROGRESS.md`. Do not build environments or a replacement viewer. Read upstream MJCF + JSON; preserve unknown/assumed scale and confidence. Never equate route planning with learned locomotion or simulation with hardware commissioning.

Ownership: scene/planning agent -> `scene.py`, `planning.py`, their tests; policy agent -> `policies/g1.py`, `scripts/fetch_g1.py`, policy tests; visualization agent -> `web/`, visualization handoff; root -> packaging, orchestration/evaluation, navigation/optimization, integration tests/docs. You are not alone; accommodate concurrent edits and do not revert others.

After every saved source edit, record a compact checkpoint before the next edit: goal/constraints, changed paths/intent, contracts/decisions, exact check outcomes, blockers, next step. Use the existing PROGRESS note; per-agent state may be recorded in a concise owned section or tool checkpoint to avoid races. There is no callable manual compaction tool in this session; do not claim compaction occurred.

Build/check commands (once package exists): `uv sync --extra dev`, `uv run pytest`, `uv run morpheus-policy --help`. Record real-run commands and outcomes. No `.codegraph/` exists initially. Reach for CodeGraph before code search if one is later present; do not create its index automatically.

Skill discovery initially failed because provider cache roots changed; `skills.include_instructions=true` was confirmed in the host config and the supplied current catalog was reloaded as fallback. Do not alter the user's global skill registry for this task.
