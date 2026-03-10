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
  - Package: `com.ivanmurzak.unity.mcp` **v0.51.5**
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
   - **2k–8k particles** in Editor is acceptable
   - Must avoid O(n^2) by neighbor search via spatial hash/uniform grid
5) Controls:
   - `Space` pause/resume
   - `RightArrow` single-step when paused
   - `R` reset (respawn)

### 3.2 Required file layout (must match)
Create these assets (paths are strict):
- `Assets/Scripts/Fluid2D/FluidSim2D.cs`
- `Assets/Scripts/Fluid2D/SpatialHash2D.cs`
- `Assets/Scripts/Fluid2D/FluidKernels2D.cs`
- `Assets/Scripts/Fluid2D/ParticleRenderer2D.cs`
- `Assets/Shaders/Particles2D_UnlitCircle.shader`
- `Assets/Scenes/Fluid2D_Demo.unity` (Demo scene)

---

## 4) Implementation Requirements (CPU-first, stable-first)

### 4.1 Spatial Hash / Uniform Grid (GC-free)
- Use cell size ≈ `smoothingRadius` (or a constant multiple).
- Neighbor search checks **3x3** surrounding cells.
- Avoid allocations per frame:
  - Reuse `List<int>` / arrays
  - No LINQ
- Recommended structure:
  - `Dictionary<int,int> cellHead` mapping cellKey -> headIndex
  - `int[] next` as linked list within each bucket
  - Build buckets each step, then query neighbors by iterating lists.

### 4.2 Fluid model (recommend near-density + near-pressure)
Goal: stability and “liquid-like” behavior before accuracy.

Recommended approach (2D):
- For neighbor distance `r < h`:
  - `q = 1 - r/h` clamped to [0,1]
  - `density += q^2`
  - `nearDensity += q^3`
- Pressures:
  - `pressure = (density - targetDensity) * pressureMultiplier`
  - `nearPressure = nearDensity * nearPressureMultiplier`
- Pressure force:
  - direction `dir = (xi - xj) / max(r, eps)`
  - magnitude scales with `pressure*q + nearPressure*q^2`
  - Must include `eps` to prevent blow-ups at tiny r
- Viscosity:
  - use `(vj - vi) * q * viscosityStrength` style smoothing
- Integration:
  - semi-implicit Euler
  - optional substeps via `iterationsPerFrame` (1–4) for stability

### 4.3 Collisions
- Bounds: AABB centered at (0,0) with half-extents = `boundsSize * 0.5`
- Obstacle: AABB with half-extents = `obstacleSize * 0.5`
- On collision:
  - clamp position to boundary surface
  - reflect velocity component, multiply by `collisionDamping` (0.9–0.98 typical)

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
- `numParticles = 4000`
- `smoothingRadius = 0.2` (world scale dependent)
- `gravity = -9.8`
- `collisionDamping = 0.95`
- `interactionRadius` noticeable (e.g., 0.8–1.5 range) and `interactionStrength` noticeable
- Provide reasonable starting values for:
  - `targetDensity`
  - `pressureMultiplier`
  - `nearPressureMultiplier`
  - `viscosityStrength`
- `iterationsPerFrame = 2` if needed for stability

Also add brief comments in code explaining:
- increasing pressure multipliers -> stiffer but risk explosion
- increasing viscosity -> thicker/slower flow
- decreasing dt / increasing substeps -> more stable

---

## 7) Demo Scene Contract (must be reproducible)
Create `Assets/Scenes/Fluid2D_Demo.unity` via MCP:
- Root GameObject: `FluidSim2D_Root`
  - components: `FluidSim2D`, `ParticleRenderer2D`
- Camera:
  - Orthographic, positioned to view bounds
- Material:
  - `Assets/Materials/Particles2D.mat` using `Particles2D_UnlitCircle.shader`
- Ensure Play Mode shows fluid immediately without manual wiring.

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

For each milestone, chat response must include:
- changed/created asset paths
- scene hierarchy changes
- key Inspector defaults
- console errors/warnings (if any) + fixes applied
- NO full code dumps