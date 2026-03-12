# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

---

## 0) Operating Contract (READ FIRST)

### Use Unity MCP for Unity assets
- Use MCP tools (`mcp__ai-game-developer__*`) to create/modify **Scenes, GameObjects, Components, Materials, Shaders**.
- Do **NOT** hand-edit `.unity` / `.asset` files in text.

### Low-token chat output (IMPORTANT)
- **Do NOT print full source files** in chat.
- Allowed in chat:
  - Asset/script paths created/modified
  - Scene hierarchy changes
  - Inspector parameter values (key fields)
  - Small snippets/diffs **≤ 30 lines total** only when necessary (e.g., core formula loop, hash function)
- Everything else must be written directly into the project via MCP tools and then compiled.

### After every meaningful change
1) `mcp__ai-game-developer__assets-refresh`
2) `mcp__ai-game-developer__console-get-logs` and report errors/warnings
3) If runtime validation is needed: enter Play Mode (via MCP if available) and sanity-check behavior.

---

## 1) Project Overview

- Unity version: **2022.3.34f1**
- Project goal: **2D CPU particle fluid simulation (MVP)** inspired by SebLague/Fluid-Sim
- Unity MCP plugin:
  - Package: `com.ivanmurzak.unity.mcp` **v0.51.6**
  - MCP server: `localhost:53540`
- `.mcp.json` in repo root configures Claude Code to connect to the running Unity Editor MCP server.

---

## 2) MCP Workflow Cheatsheet

Prefer these tools:
- Create/update scripts: `mcp__ai-game-developer__script-update-or-create`
- Refresh/compile: `mcp__ai-game-developer__assets-refresh`
- Inspect console: `mcp__ai-game-developer__console-get-logs`
- Scene edits (objects/components): use relevant `mcp__ai-game-developer__scene-*` / `mcp__ai-game-developer__gameobject-*` tools
- One-off editor-side execution: `mcp__ai-game-developer__script-execute` (Roslyn)

---

## 3) MVP Spec — 2D Fluid Particles (CPU)

### 3.1 Acceptance Criteria (must pass)
1) Press Play: particles fall under gravity, pile up, and flow like liquid.
2) Collisions:
   - World bounds: AABB centered at origin (`boundsSize`)
   - One obstacle: AABB (`obstacleCentre`, `obstacleSize`)
   - Bounce with damping (`collisionDamping`)
3) Mouse interaction:
   - **LMB** attract, **RMB** repel
   - `interactionRadius`, `interactionStrength` exposed
4) Performance:
   - improve gradually, from 10k to 1M+
5) Controls:
   - `Space` pause/resume
   - `RightArrow` single-step when paused
   - `R` reset (respawn)

---

## 4) Implementation Requirements 
//

---

## 5) Rendering & Debug

### 5.1 Rendering (fast + simple)
- `Graphics.DrawMeshInstanced` with a quad mesh.
- Batch size 1023; reuse matrix/color arrays each frame.
- Shader:
  - Unlit, transparent
  - Fragment alpha masks to a circle (soft edge ok)
- Optional debug:
  - color by speed (helps tuning)

### 5.2 Gizmos / HUD
- `OnDrawGizmos` draw:
  - bounds AABB
  - obstacle AABB
  - interaction radius circle at mouse world position (when pressed)
- Optional small on-screen stats (particle count, dt, fps) if inexpensive.

---

## 6) Default Parameters (stable baseline)
Provide defaults in `FluidSim2D` that should “look like fluid” out of the box:

Also add brief comments in code explaining:
- increasing pressure multipliers -> stiffer but risk explosion
- increasing viscosity -> thicker/slower flow
- decreasing dt / increasing substeps -> more stable

---

## 8) Tests (existing scaffold)
Tests are in:
- `Assets/com.IvanMurzak/AI Game Dev Installer/Tests/`
Run via:
- Unity: Window → General → Test Runner → Run All
- MCP: `mcp__ai-game-developer__tests-run`

---

## 9) Stepwise Delivery Plan (recommended)
Work in milestones, validating compilation and behavior after each:

1) Scene + shader/material + script skeletons compile
2) SpatialHash2D implemented and verified (no allocations)
3) Density/pressure/viscosity + integration produce stable motion
4) Collisions (bounds + obstacle) correct
5) Mouse interaction + controls (pause/step/reset)
6) Rendering correctness + Gizmos + light debug stats (optional)

---

## 10) Git & Version Control
Purpose: keep experiments isolated and history clean.

### 10.1 Core Rules
- `main` must always be runnable and stable.
- Do not develop directly on `main`.
- Use small, focused commits.
- Do not commit generated or temporary files.

---

### 10.2 Branching
- `feature/<name>` — new functionality
- `exp/<name>` — experimental algorithm variants
- `fix/<name>` — bug fixes

Rules:
- Each major algorithm change must use its own branch.
- Different solution paths must use separate `exp/*` branches.
- Only merge validated solutions into `main`.
- Delete unused experiment branches after conclusion.

---

### 10.3 Tags
Tag stable checkpoints:

```bash
git tag -a v0.1.0 -m "Stable CPU baseline"
git push origin v0.1.0
```
- Create a tag when:
  - Stability significantly improves
  - Performance milestone is reached
  - Architecture changes substantially



For each milestone, chat response must include:
- changed/created asset paths
- scene hierarchy changes
- key Inspector defaults
- console errors/warnings (if any) + fixes applied
- NO full code dumps