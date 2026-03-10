using UnityEngine;
using Unity.Mathematics;
using Unity.Jobs;
using Unity.Collections;

namespace Fluid2D
{
    public class FluidSim2D : MonoBehaviour
    {
        [Header("Simulation")]
        public int numParticles = 2000;
        public float smoothingRadius = 0.2f;
        // Decrease dt or increase iterationsPerFrame for more stability
        public int iterationsPerFrame = 3;

        [Header("Gravity")]
        public float gravity = -9.8f;

        [Header("Pressure")]
        // Higher pressureMultiplier -> stiffer fluid, risks explosion
        public float targetDensity         = 2.75f;
        public float pressureMultiplier    = 80f;
        // Higher nearPressureMultiplier -> stronger short-range repulsion
        public float nearPressureMultiplier = 15f;

        [Header("Viscosity")]
        // Higher viscosityStrength -> thicker / slower flow
        public float viscosityStrength = 0.1f;

        [Header("Collision")]
        public Vector2 boundsSize      = new Vector2(16f, 9f);
        public float   collisionDamping = 0.4f;
        // Set obstacleSize to (0,0) to disable
        public Vector2 obstacleCentre = new Vector2(0f, -1.5f);
        public Vector2 obstacleSize   = Vector2.zero;

        [Header("Interaction")]
        public float interactionRadius   = 1.2f;
        public float interactionStrength = 10f;

        // Public NativeArrays — read by ParticleRenderer2D
        public NativeArray<float2> positionsNative;
        public NativeArray<float2> velocitiesNative;
        public int particleCount => numParticles;

        // Internal simulation data
        NativeArray<float2> _predicted;
        NativeArray<float>  _densities;
        NativeArray<float>  _nearDensities;
        NativeArray<float2> _velocityDelta;
        NativeArray<int>    _gridHead;
        NativeArray<int>    _gridNext;

        // Grid parameters (computed in Spawn)
        int    _gridW, _gridH;
        float2 _gridOrigin;
        float  _invCellSize;

        bool _allocated;
        bool _paused;

        // Mouse state (set in Update, used in Step)
        Vector2 _mouseWorld;
        bool    _mouseLeft;
        bool    _mouseRight;

        float _fpsSmoothed = 60f;

        void Start() => Spawn();
        void Update() => HandleInput();

        void FixedUpdate()
        {
            if (_paused || !_allocated) return;
            float dt = Time.fixedDeltaTime / iterationsPerFrame;
            for (int s = 0; s < iterationsPerFrame; s++) Step(dt);
        }

        void OnDestroy() => DisposeArrays();

        // ── Spawn ──────────────────────────────────────────────────────────────

        public void Spawn()
        {
            DisposeArrays();
            int n = numParticles;
            var alloc = Allocator.Persistent;

            positionsNative  = new NativeArray<float2>(n, alloc);
            velocitiesNative = new NativeArray<float2>(n, alloc);
            _predicted       = new NativeArray<float2>(n, alloc);
            _densities       = new NativeArray<float>  (n, alloc);
            _nearDensities   = new NativeArray<float>  (n, alloc);
            _velocityDelta   = new NativeArray<float2>(n, alloc);
            _gridNext        = new NativeArray<int>    (n, alloc);

            // Fixed-array grid sized to world bounds with padding
            float cellSize  = smoothingRadius;
            _invCellSize    = 1f / cellSize;
            int pad         = 4;
            _gridW          = Mathf.CeilToInt(boundsSize.x / cellSize) + pad * 2;
            _gridH          = Mathf.CeilToInt(boundsSize.y / cellSize) + pad * 2;
            _gridOrigin     = new float2(
                -boundsSize.x * 0.5f - pad * cellSize,
                -boundsSize.y * 0.5f - pad * cellSize);
            _gridHead = new NativeArray<int>(_gridW * _gridH, alloc);
            for (int i = 0; i < _gridHead.Length; i++) _gridHead[i] = -1;

            _allocated = true;

            // Spawn grid that always fits within bounds regardless of particle count
            float innerW = boundsSize.x - smoothingRadius * 2f;
            float innerH = boundsSize.y - smoothingRadius * 2f;
            // Match aspect ratio so grid fills bounds proportionally
            int   cols   = Mathf.Max(1, Mathf.RoundToInt(Mathf.Sqrt(n * innerW / Mathf.Max(innerH, 0.001f))));
            int   rows   = Mathf.CeilToInt((float)n / cols);
            // Derive spacing from whichever axis is tighter
            float sX     = cols > 1 ? innerW / (cols - 1) : innerW;
            float sY     = rows > 1 ? innerH / (rows - 1) : innerH;
            float spacing = Mathf.Max(Mathf.Min(sX, sY), 0.001f);
            float totalW  = (cols - 1) * spacing;
            float totalH  = (rows - 1) * spacing;
            float2 origin = new float2(-totalW * 0.5f, totalH * 0.5f);

            for (int i = 0; i < n; i++)
            {
                var jitter = (Vector2)UnityEngine.Random.insideUnitCircle * spacing * 0.05f;
                positionsNative[i]  = origin + new float2(
                    (i % cols) * spacing + jitter.x,
                    -(i / cols) * spacing + jitter.y);
                velocitiesNative[i] = new float2(0f, 0f);
            }
        }

        // ── Physics step (schedules 5 Burst jobs) ─────────────────────────────

        void Step(float dt)
        {
            int    n           = numParticles;
            float2 gridOrigin  = _gridOrigin;
            float  invCellSize = _invCellSize;
            int    gridW       = _gridW;
            int    gridH       = _gridH;

            var h1 = new GravityPredictJob
            {
                dt         = dt,
                gravity    = gravity,
                velocities = velocitiesNative,
                positions  = positionsNative,
                predicted  = _predicted,
            }.Schedule(n, 64);

            var h2 = new BuildGridJob
            {
                predicted   = _predicted,
                gridHead    = _gridHead,
                gridNext    = _gridNext,
                count       = n,
                gridW       = gridW,
                gridH       = gridH,
                gridOrigin  = gridOrigin,
                invCellSize = invCellSize,
            }.Schedule(h1);

            var h3 = new DensityJob
            {
                predicted     = _predicted,
                densities     = _densities,
                nearDensities = _nearDensities,
                gridHead      = _gridHead,
                gridNext      = _gridNext,
                gridW         = gridW,
                gridH         = gridH,
                gridOrigin    = gridOrigin,
                invCellSize   = invCellSize,
                h             = smoothingRadius,
            }.Schedule(n, 64, h2);

            var h4 = new ForceJob
            {
                predicted              = _predicted,
                densities              = _densities,
                nearDensities          = _nearDensities,
                velocities             = velocitiesNative,
                velocityDelta          = _velocityDelta,
                gridHead               = _gridHead,
                gridNext               = _gridNext,
                gridW                  = gridW,
                gridH                  = gridH,
                gridOrigin             = gridOrigin,
                invCellSize            = invCellSize,
                h                      = smoothingRadius,
                targetDensity          = targetDensity,
                pressureMultiplier     = pressureMultiplier,
                nearPressureMultiplier = nearPressureMultiplier,
                viscosityStrength      = viscosityStrength,
                dt                     = dt,
            }.Schedule(n, 64, h3);

            var h5 = new IntegrateJob
            {
                velocities       = velocitiesNative,
                positions        = positionsNative,
                velocityDelta    = _velocityDelta,
                dt               = dt,
                boundsHalf       = new float2(boundsSize.x * 0.5f, boundsSize.y * 0.5f),
                collisionDamping = collisionDamping,
                smoothingRadius  = smoothingRadius,
                obstacleCentre   = new float2(obstacleCentre.x, obstacleCentre.y),
                obstacleHalf     = new float2(obstacleSize.x * 0.5f, obstacleSize.y * 0.5f),
                hasObstacle      = obstacleSize.sqrMagnitude > 0.0001f,
            }.Schedule(n, 64, h4);

            h5.Complete();

            // Mouse interaction on main thread after jobs complete (O(n), not a bottleneck)
            ApplyMouseInteraction(dt);
        }

        // ── Mouse interaction ─────────────────────────────────────────────────

        void ApplyMouseInteraction(float dt)
        {
            if (!_mouseLeft && !_mouseRight) return;
            float  sign   = _mouseLeft ? -1f : 1f;  // LMB = attract, RMB = repel
            float  r2Max  = interactionRadius * interactionRadius;
            float2 mouseW = new float2(_mouseWorld.x, _mouseWorld.y);
            int    n      = numParticles;

            for (int i = 0; i < n; i++)
            {
                float2 diff = positionsNative[i] - mouseW;
                float  r2   = math.dot(diff, diff);
                if (r2 > r2Max || r2 < 1e-8f) continue;
                float r = math.sqrt(r2);
                float t = 1f - r / interactionRadius;
                var v = velocitiesNative[i];
                v += (sign * diff / r) * (t * interactionStrength * dt);
                velocitiesNative[i] = v;
            }
        }

        // ── Input ─────────────────────────────────────────────────────────────

        void HandleInput()
        {
            if (Input.GetKeyDown(KeyCode.Space))    _paused = !_paused;
            if (Input.GetKeyDown(KeyCode.R))        Spawn();
            if (_paused && Input.GetKeyDown(KeyCode.RightArrow))
            {
                float dt = Time.fixedDeltaTime / iterationsPerFrame;
                for (int s = 0; s < iterationsPerFrame; s++) Step(dt);
            }

            _mouseLeft  = Input.GetMouseButton(0);
            _mouseRight = Input.GetMouseButton(1);
            if ((_mouseLeft || _mouseRight) && Camera.main != null)
            {
                Vector3 mp = Camera.main.ScreenToWorldPoint(Input.mousePosition);
                _mouseWorld = new Vector2(mp.x, mp.y);
            }
        }

        // ── Cleanup ────────────────────────────────────────────────────────────

        void DisposeArrays()
        {
            if (!_allocated) return;
            positionsNative.Dispose();
            velocitiesNative.Dispose();
            _predicted.Dispose();
            _densities.Dispose();
            _nearDensities.Dispose();
            _velocityDelta.Dispose();
            _gridNext.Dispose();
            _gridHead.Dispose();
            _allocated = false;
        }

        // ── Gizmos ────────────────────────────────────────────────────────────

        void OnDrawGizmos()
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireCube(Vector3.zero, new Vector3(boundsSize.x, boundsSize.y, 0));

            if (obstacleSize.sqrMagnitude > 0.0001f)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawWireCube(
                    new Vector3(obstacleCentre.x, obstacleCentre.y, 0),
                    new Vector3(obstacleSize.x, obstacleSize.y, 0));
            }

            if (Application.isPlaying && (_mouseLeft || _mouseRight))
            {
                Gizmos.color = _mouseLeft ? new Color(0f, 1f, 0.5f, 0.4f) : new Color(1f, 0.4f, 0f, 0.4f);
                Gizmos.DrawWireSphere(new Vector3(_mouseWorld.x, _mouseWorld.y, 0), interactionRadius);
            }
        }

        // ── HUD ───────────────────────────────────────────────────────────────

        void OnGUI()
        {
            if (!_allocated) return;
            _fpsSmoothed = Mathf.Lerp(_fpsSmoothed, 1f / Mathf.Max(Time.deltaTime, 1e-4f), 0.05f);
            var style = new GUIStyle(GUI.skin.label) { fontSize = 14 };
            style.normal.textColor = Color.white;
            float y = 8f;
            GUI.Label(new Rect(8, y, 260, 22), $"Particles : {numParticles}",    style); y += 20;
            GUI.Label(new Rect(8, y, 260, 22), $"FPS       : {_fpsSmoothed:F1}", style); y += 20;
            GUI.Label(new Rect(8, y, 260, 22), $"Substeps  : {iterationsPerFrame}", style); y += 24;
            GUI.Label(new Rect(8, y, 260, 22), $"[Space] {(_paused ? "Resume" : "Pause")}  [R] Reset  [→] Step", style);
        }
    }
}
