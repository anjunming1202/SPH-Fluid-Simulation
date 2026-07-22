# GPU Fluid Simulation (2D SPH)

Real-time 2D fluid simulation with **100K particles at 200+ FPS** (up to 500K at 60 FPS) using GPU-accelerated Smoothed Particle Hydrodynamics in Unity.

![Fluid interaction demo](docs/demo_interaction.gif)

<!-- Full demo video: [YouTube](TODO) -->

## Features

- **GPU Compute Shader** pipeline — zero CPU readback, all physics on GPU
- **Wendland C2 kernel** — eliminates pairing instability common in cubic spline SPH
- **O(N) Counting Sort** with Blelloch prefix scan — replaces O(N log²N) bitonic sort
- **Akinci 2012 boundary particles** — physically correct wall density compensation
- **CFL-adaptive timestep** — automatic stability without manual dt tuning
- **Mouse interaction** — LMB attract / RMB repel particles in real-time
- **Density clamping** — prevents explosion from transient particle overlaps

### Obstacle interaction

![Obstacle pressure demo](docs/demo_obstacle.gif)

*Obstacle compresses fluid against the wall, building pressure. When released, high-pressure fluid erupts from the gap — demonstrating stable pressure dynamics even under extreme compression.*

## Algorithm Overview

### SPH Density & Pressure (WCSPH)

Each particle's density is computed by summing Wendland C2 kernel contributions from neighbors within the smoothing radius. Pressure follows the weakly compressible equation of state: `P = k(ρ - ρ₀)`, with negative-pressure cohesion for surface tension effects. A near-density repulsion term (short-range polynomial kernel) prevents particle collapse.

### Spatial Hashing — O(N) Counting Sort

Particles are assigned to grid cells based on position. A GPU Blelloch prefix scan computes cell offsets in 6 dispatches, enabling sorted-order neighbor lookups. This replaces the earlier O(N log²N) bitonic sort that became the bottleneck at 500K+ particles.

### Boundary Handling — Akinci 2012

Wall boundaries use analytical boundary particles ([Akinci et al. 2012](https://cg.informatik.uni-freiburg.de/publications/2012_SIGGRAPHASIA_rigidFluidCoupling.pdf)) instead of ghost forces. Each wall face contributes virtual particle density `ψ_b = ρ₀ / Σ W(k·d)` to nearby fluid particles, correcting the kernel deficiency at boundaries and providing physically based repulsion + no-slip viscosity.

### CFL-Adaptive Timestep

The timestep is clamped by `dt_CFL = 0.4h / c_s` where `c_s = √(k/ρ₀)` is the speed of sound. This automatically prevents instability when pressure stiffness or velocity increases.

## Controls

| Key | Action |
|-----|--------|
| **LMB** | Attract particles toward cursor |
| **RMB** | Repel particles from cursor |
| **Space** | Pause / Resume |
| **Right Arrow** | Single-step (when paused) |
| **R** | Reset simulation |

## Parameters

<details>
<summary><b>Inspector parameters reference</b></summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numParticles` | 100,000 | Total particle count |
| `substeps` | 1 | Full substeps per frame (rebuilds sort grid each) |
| `smoothingRadius` | 0.05 | SPH kernel radius `h` |
| `targetDensity` | 1.5 | Rest density `ρ₀` (auto-computed when `autoRestDensity` is on) |
| `pressureMultiplier` | 400 | Pressure stiffness `k` — higher = stiffer, risk of instability |
| `cohesionPressure` | 400 | Negative-pressure attraction strength (surface tension approximation) |
| `nearPressureMultiplier` | 10 | Short-range repulsion — prevents particle collapse |
| `viscosityStrength` | 40 | Higher = thicker, slower flow |
| `collisionDamping` | 0.4 | Velocity retention on wall bounce (0 = full absorb, 1 = elastic) |
| `maxDensityRatio` | 4 | Density clamp as multiple of `ρ₀` — prevents explosion from overlaps |
| `maxVelocity` | 200 | Emergency speed cap |
| `boundsSize` | (16, 9) | Simulation domain size |
| `interactionRadius` | 1.5 | Mouse interaction radius |
| `interactionStrength` | 50 | Mouse force strength |

</details>

## Performance

Tested on RTX 4070 Ti Super, Unity 2022.3:

| Particles | FPS | Notes |
|-----------|-----|-------|
| 100K | 200+ | Smooth, best visual quality |
| 500K | ~60 | Upper limit for real-time on this GPU |
| 700K | ~30 | Playable but not smooth |

Bottleneck at high particle counts is `ComputeDensity` + `ComputeForce` (O(N*k) neighbor interactions). Counting Sort overhead is negligible.

## Getting Started

**Requirements:** Unity 2022.3 LTS

```bash
git clone https://github.com/anjunming1202/SPH-Fluid-Simulation.git
```

1. Open the project in Unity 2022.3
2. Open `Assets/Scenes/FluidGPU_Demo.unity`
3. Press Play

## Project Structure

```
Assets/
├── Scripts/Fluid2D/
│   └── FluidSimGPU2D.cs          # Main simulation controller
├── Shaders/
│   ├── FluidCompute2D.compute     # SPH kernels (Density/Force/Integrate)
│   ├── CountingSort.compute       # O(N) GPU sorting (Blelloch scan)
│   ├── BitonicSort.compute        # Legacy O(N log²N) sort (fallback)
│   └── FluidRenderGPU.shader      # Instanced particle rendering
└── Scenes/
    └── FluidGPU_Demo.unity        # Demo scene (100K particles)
```

## Future Work

- **Surface tension** — Current cohesion pressure is a rough approximation. A proper CSF (Continuum Surface Force) or pairwise surface tension model would produce visible meniscus and droplet formation.
- **Incompressible SPH** — WCSPH has noticeable compressibility. Upgrading to IISPH or DFSPH would improve volume conservation, enabling classic demonstrations like U-tube equilibrium.
- **Auto-scaling parameters** — Changing particle count currently requires manual re-tuning. A scaling law that maps `(N, bounds, h)` to stable `(ρ₀, k, viscosity)` would make the simulation resolution-independent.

## License

[MIT](LICENSE)
