// FluidSimGPU2D.cs
// GPU-driven 2D SPH fluid simulation.
// All particle data lives in ComputeBuffers — no CPU readback.
// Rendering: DrawMeshInstancedIndirect reads _RenderData directly on GPU.
//
// Phase 2: Bitonic Sort grid. Targets 500k particles.
//   Sorted particle array replaces atomic bucket; no MAX_PER_CELL limit.
//
// Setup:
//   1. Assign computeShader -> FluidCompute2D.compute
//   2. Particle material + mesh auto-created if left null.
//   3. Press Play. Controls: Space=pause, RightArrow=step, R=reset.

using UnityEngine;

[AddComponentMenu("Fluid/FluidSimGPU2D")]
public class FluidSimGPU2D : MonoBehaviour
{
    // ── Simulation parameters ────────────────────────────────────────────────
    [Header("Simulation")]
    [Tooltip("Total particle count. 100k is Phase 1 target.")]
    public int numParticles = 100000;

    [Tooltip("Sub-steps per frame. More = stable, slower.")]
    [Range(1, 5)]
    public int substeps = 3;

    public float maxDt = 0.01f;

    public float gravity              = -9.8f;
    public float smoothingRadius      = 0.2f;

    [Tooltip("When true, targetDensity is auto-computed from particle count and bounds so the simulation stays stable regardless of numParticles or boundsSize.")]
    public bool  autoRestDensity      = true;
    public float targetDensity        = 2.75f;

    public float pressureMultiplier   = 80f;

    [Tooltip("Max suction at low density (surface tension). 0=powder, ~10=water, >20=jelly.")]
    public float cohesionPressure     = 10f;

    public float nearPressureMultiplier = 15f;

    [Tooltip("Higher = thicker/slower flow")]
    public float viscosityStrength    = 0.1f;

    [Range(0f, 1f)]
    public float collisionDamping     = 0.4f;

    [Tooltip("Emergency speed cap. CFL-adaptive dt is the real stability mechanism; this only catches extremes.")]
    public float maxVelocity          = 200f;

    // ── World ────────────────────────────────────────────────────────────────
    [Header("World")]
    public Vector2 boundsSize      = new Vector2(16f, 9f);
    public Vector2 obstacleCentre  = Vector2.zero;
    public Vector2 obstacleSize    = Vector2.zero;

    // ── Mouse interaction ────────────────────────────────────────────────────
    [Header("Mouse")]
    public float interactionRadius   = 1.5f;
    public float interactionStrength = 5f;

    // ── Rendering ────────────────────────────────────────────────────────────
    [Header("Rendering")]
    [Tooltip("Auto-created quad if left null")]
    public Mesh particleMesh;

    [Tooltip("Auto-created from FluidRenderGPU shader if left null")]
    public Material particleMaterial;

    public float particleSize = 0.08f;

    // ── Debug tools ──────────────────────────────────────────────────────────
    [Header("Debug")]
    [Tooltip("Hold Alt to show density at cursor. 1 = density color mode")]
    public bool debugDensityColor = false;

    [Tooltip("Density value that maps to full red in density color mode")]
    public float debugDensityScale = 10f;

    // ── Compute shaders (required) ───────────────────────────────────────────
    [Header("References")]
    public ComputeShader computeShader;
    public ComputeShader bitonicShader;

    // ── ComputeBuffers ───────────────────────────────────────────────────────
    ComputeBuffer _positions, _velocities, _predicted;
    ComputeBuffer _densities, _nearDensities, _velocityDelta;
    ComputeBuffer _sortKeys, _sortIndices;  // [paddedN] sorted by cell key
    ComputeBuffer _cellStart, _cellEnd;     // [numCells] index range per cell
    ComputeBuffer _renderData;
    ComputeBuffer _argsBuffer;

    // ── Grid ─────────────────────────────────────────────────────────────────
    int   _gridW, _gridH, _numCells;
    float _cellSize;
    int   _paddedN;  // next power of 2 >= numParticles (for Bitonic Sort)

    // ── Kernel IDs ────────────────────────────────────────────────────────────
    int _kClearCellStarts, _kGravityPredict, _kComputeKeys, _kCalcCellStarts;
    int _kComputeDensity, _kComputeForce, _kIntegrate, _kBuildRenderData;
    int _kQueryDensityAtPos;
    int _kBitonicStep;

    // ── Debug query ───────────────────────────────────────────────────────────
    ComputeBuffer _queryBuffer;     // [0]=density [1]=nearDensity
    float[]       _queryData = new float[2];
    Vector2       _queryWorldPos;
    bool          _queryActive;
    Material      _glMat;           // for GL circle overlay

    // ── Runtime state ─────────────────────────────────────────────────────────
    bool   _paused;
    bool   _initialized;
    Camera _cam;

    // ─────────────────────────────────────────────────────────────────────────
    void Start()
    {
        _cam = Camera.main;
        Init();
    }

    void Init()
    {
        ReleaseBuffers();

        // Validate
        if (computeShader == null)
        {
            Debug.LogError("[FluidSimGPU2D] computeShader is not assigned!");
            return;
        }

        // Auto-create mesh / material if missing
        if (particleMesh == null)
            particleMesh = CreateQuadMesh();

        if (particleMaterial == null)
        {
            var s = Shader.Find("Custom/FluidRenderGPU");
            if (s != null)
                particleMaterial = new Material(s);
            else
                Debug.LogError("[FluidSimGPU2D] Shader 'Custom/FluidRenderGPU' not found. Assign particleMaterial manually.");
        }

        // Grid dimensions — cell size = smoothingRadius
        _cellSize = smoothingRadius;
        _gridW    = Mathf.CeilToInt(boundsSize.x / _cellSize) + 2;
        _gridH    = Mathf.CeilToInt(boundsSize.y / _cellSize) + 2;
        _numCells = _gridW * _gridH;

        // Auto rest-density: derived from q² kernel integral over uniform distribution.
        // ρ_rest = N × π × h² / (3 × area) — keeps pressure near-zero at equilibrium
        // regardless of particle count, bounds size, or smoothing radius.
        if (autoRestDensity)
        {
            float area = boundsSize.x * boundsSize.y;
            targetDensity = numParticles * Mathf.PI * smoothingRadius * smoothingRadius / (3f * area);
            Debug.Log($"[FluidSimGPU2D] autoRestDensity → targetDensity = {targetDensity:F4} (N={numParticles}, h={smoothingRadius}, area={area:F1})");
        }

        // Allocate GPU buffers
        _paddedN        = Mathf.NextPowerOfTwo(numParticles);
        _positions      = new ComputeBuffer(numParticles, 2 * sizeof(float));
        _velocities     = new ComputeBuffer(numParticles, 2 * sizeof(float));
        _predicted      = new ComputeBuffer(numParticles, 2 * sizeof(float));
        _densities      = new ComputeBuffer(numParticles,     sizeof(float));
        _nearDensities  = new ComputeBuffer(numParticles,     sizeof(float));
        _velocityDelta  = new ComputeBuffer(numParticles, 2 * sizeof(float));
        _sortKeys       = new ComputeBuffer(_paddedN,         sizeof(uint));
        _sortIndices    = new ComputeBuffer(_paddedN,         sizeof(uint));
        _cellStart      = new ComputeBuffer(_numCells,        sizeof(uint));
        _cellEnd        = new ComputeBuffer(_numCells,        sizeof(uint));
        _renderData     = new ComputeBuffer(numParticles, 4 * sizeof(float));
        _queryBuffer    = new ComputeBuffer(2,                sizeof(float));

        // Indirect args: indexCount, instanceCount, startIndex, baseVertex, startInstance
        _argsBuffer = new ComputeBuffer(5, sizeof(uint), ComputeBufferType.IndirectArguments);
        var args = new uint[5];
        args[0] = (uint)particleMesh.GetIndexCount(0);
        args[1] = (uint)numParticles;
        args[2] = (uint)particleMesh.GetIndexStart(0);
        args[3] = (uint)particleMesh.GetBaseVertex(0);
        args[4] = 0u;
        _argsBuffer.SetData(args);

        // Kernel IDs — FluidCompute2D.compute
        _kClearCellStarts    = computeShader.FindKernel("ClearCellStarts");
        _kGravityPredict     = computeShader.FindKernel("GravityPredict");
        _kComputeKeys        = computeShader.FindKernel("ComputeKeys");
        _kCalcCellStarts     = computeShader.FindKernel("CalcCellStarts");
        _kComputeDensity     = computeShader.FindKernel("ComputeDensity");
        _kComputeForce       = computeShader.FindKernel("ComputeForce");
        _kIntegrate          = computeShader.FindKernel("Integrate");
        _kBuildRenderData    = computeShader.FindKernel("BuildRenderData");
        _kQueryDensityAtPos  = computeShader.FindKernel("QueryDensityAtPos");

        // BitonicSort.compute
        if (bitonicShader != null)
        {
            _kBitonicStep = bitonicShader.FindKernel("BitonicStep");
            bitonicShader.SetInt("_SortN", _paddedN);
            bitonicShader.SetBuffer(_kBitonicStep, "_SortKeys",    _sortKeys);
            bitonicShader.SetBuffer(_kBitonicStep, "_SortIndices", _sortIndices);
        }
        else
        {
            Debug.LogError("[FluidSimGPU2D] bitonicShader not assigned! Assign BitonicSort.compute in the Inspector.");
        }

        // Spawn particles in a grid
        SpawnParticles();

        // Bind buffers + static uniforms
        BindBuffers();
        SetStaticUniforms();

        // Pass render buffer to material
        if (particleMaterial != null)
        {
            particleMaterial.SetBuffer("_RenderData",   _renderData);
            particleMaterial.SetFloat ("_ParticleSize", particleSize);
        }

        _initialized = true;
        /*_paused      = false;*/
        Debug.Log($"[FluidSimGPU2D] Initialized: {numParticles} particles (paddedN={_paddedN}), grid {_gridW}×{_gridH} ({_numCells} cells), targetDensity={targetDensity:F4}, bitonicDispatches={BitonicDispatchCount(_paddedN)}");
    }

    // ── Particle spawn ────────────────────────────────────────────────────────
    void SpawnParticles()
    {
        var pos = new Vector2[numParticles];
        var vel = new Vector2[numParticles];

        float spacing = smoothingRadius * 0.9f;
        float bEps    = smoothingRadius * 0.15f;

        // Fit columns within bounds width — no particle ever gets clamped
        int cols = Mathf.Max(1, Mathf.FloorToInt((boundsSize.x - bEps * 2f) / spacing));

        // Start from top of bounds, stack downward
        float startX = -(cols - 1) * spacing * 0.5f;
        float startY =  boundsSize.y * 0.5f - bEps - spacing * 0.5f;

        for (int i = 0; i < numParticles; i++)
        {
            int   col = i % cols;
            int   row = i / cols;
            pos[i] = new Vector2(startX + col * spacing,
                                  startY - row * spacing);
            vel[i] = Vector2.zero;
        }

        _positions.SetData(pos);
        _velocities.SetData(vel);
    }

    // ── Buffer binding ────────────────────────────────────────────────────────
    void BindBuffers()
    {
        int[] kernels = {
            _kClearCellStarts, _kGravityPredict, _kComputeKeys, _kCalcCellStarts,
            _kComputeDensity, _kComputeForce, _kIntegrate,
            _kBuildRenderData, _kQueryDensityAtPos
        };

        foreach (int k in kernels)
        {
            computeShader.SetBuffer(k, "_Positions",     _positions);
            computeShader.SetBuffer(k, "_Velocities",    _velocities);
            computeShader.SetBuffer(k, "_Predicted",     _predicted);
            computeShader.SetBuffer(k, "_Densities",     _densities);
            computeShader.SetBuffer(k, "_NearDensities", _nearDensities);
            computeShader.SetBuffer(k, "_VelocityDelta", _velocityDelta);
            computeShader.SetBuffer(k, "_SortKeys",      _sortKeys);
            computeShader.SetBuffer(k, "_SortIndices",   _sortIndices);
            computeShader.SetBuffer(k, "_CellStart",     _cellStart);
            computeShader.SetBuffer(k, "_CellEnd",       _cellEnd);
            computeShader.SetBuffer(k, "_RenderData",    _renderData);
            computeShader.SetBuffer(k, "_QueryResult",   _queryBuffer);
        }
    }

    // ── Static uniforms (set once at Init) ───────────────────────────────────
    void SetStaticUniforms()
    {
        computeShader.SetInt   ("_NumParticles",          numParticles);
        computeShader.SetInt   ("_NumCells",              _numCells);
        computeShader.SetInt   ("_GridW",                 _gridW);
        computeShader.SetInt   ("_GridH",                 _gridH);
        computeShader.SetFloat ("_CellSize",              _cellSize);
        computeShader.SetFloat ("_SmoothingRadius",       smoothingRadius);
        computeShader.SetFloat ("_TargetDensity",         targetDensity);
        computeShader.SetFloat ("_PressureMultiplier",    pressureMultiplier);
        computeShader.SetFloat ("_CohesionPressure",      cohesionPressure);
        // Normalize near-pressure by targetDensity so near-repulsion force at
        // equilibrium equals (nearPressureMultiplier × 0.6) regardless of N or h.
        // Without this, nearDensity ∝ N at same h → near-pressure explodes at 500k.
        computeShader.SetFloat ("_NearPressureMultiplier",
            nearPressureMultiplier / Mathf.Max(targetDensity, 0.001f));
        computeShader.SetFloat ("_ViscosityStrength",     viscosityStrength);
        computeShader.SetFloat ("_Gravity",               gravity);
        computeShader.SetFloat ("_CollisionDamping",      collisionDamping);
        computeShader.SetInt   ("_SortN",                 _paddedN);
        computeShader.SetFloat ("_InteractionRadius",     interactionRadius);
        computeShader.SetFloat ("_InteractionStrength",   interactionStrength);
        computeShader.SetFloat ("_MaxVelocity",           maxVelocity);
        computeShader.SetInt   ("_DebugMode",             debugDensityColor ? 1 : 0);

        Vector2 bMin = -boundsSize * 0.5f;
        Vector2 bMax =  boundsSize * 0.5f;
        computeShader.SetFloats("_BoundsMin",      bMin.x, bMin.y);
        computeShader.SetFloats("_BoundsMax",      bMax.x, bMax.y);
        computeShader.SetFloats("_ObstacleCentre", obstacleCentre.x, obstacleCentre.y);
        computeShader.SetFloats("_ObstacleHalfSize", obstacleSize.x * 0.5f, obstacleSize.y * 0.5f);
    }

    // ─────────────────────────────────────────────────────────────────────────
    void Update()
    {
        if (!_initialized) return;

        // Controls
        if (Input.GetKeyDown(KeyCode.Space))      _paused = !_paused;
        if (Input.GetKeyDown(KeyCode.R))          Init();
        if (Input.GetKeyDown(KeyCode.RightArrow) && _paused) SimStep();

        if (!_paused)
            SimStep();

        // Debug density query: hold Alt to sample density at cursor
        _queryActive = Input.GetKey(KeyCode.LeftAlt) || Input.GetKey(KeyCode.RightAlt);
        if (_queryActive)
        {
            Vector3 mp = _cam.ScreenToWorldPoint(new Vector3(
                Input.mousePosition.x, Input.mousePosition.y,
                Mathf.Abs(_cam.transform.position.z)));
            _queryWorldPos = new Vector2(mp.x, mp.y);

            computeShader.SetFloats("_QueryPos", _queryWorldPos.x, _queryWorldPos.y);
            computeShader.Dispatch(_kQueryDensityAtPos, 1, 1, 1);
            _queryBuffer.GetData(_queryData);   // small sync stall — debug only
        }

        // Keep material in sync with debug mode toggle
        if (particleMaterial != null)
        {
            particleMaterial.SetInt  ("_DebugMode",  debugDensityColor ? 1 : 0);
            particleMaterial.SetFloat("_DebugScale", debugDensityScale);
            particleMaterial.SetFloat("_ParticleSize", particleSize);
        }

        // GPU-driven render (zero CPU readback)
        var bounds = new Bounds(Vector3.zero,
            new Vector3(boundsSize.x + particleSize * 2f,
                        boundsSize.y + particleSize * 2f, 1f));
        Graphics.DrawMeshInstancedIndirect(
            particleMesh, 0, particleMaterial, bounds, _argsBuffer);
    }

    // ── One simulation step (all substeps) ───────────────────────────────────
    void SimStep()
    {
        // Re-apply all uniforms so Inspector tweaks take effect immediately
        SetStaticUniforms();

        // CFL-adaptive dt: pressure wave speed c_s = sqrt(B/ρ₀).
        // dt must stay below 0.4*h/c_s so a particle can't travel more than
        // ~0.4 smoothing radii per step due to pressure alone.
        // This automatically tightens when pressureMultiplier is raised,
        // preventing explosions without needing manual dt tuning.
        float cs   = Mathf.Sqrt(pressureMultiplier / Mathf.Max(targetDensity, 0.01f));
        float dtCFL = 0.4f * smoothingRadius / cs;
        float dt   = Mathf.Min(Mathf.Min(Time.deltaTime, maxDt) / substeps, dtCFL);

        // Mouse input
        float  mouseSign  = 0f;
        Vector2 mouseWorld = Vector2.zero;
        bool lmb = Input.GetMouseButton(0);
        bool rmb = Input.GetMouseButton(1);
        if (lmb || rmb)
        {
            mouseSign = lmb ? 1f : -1f;
            Vector3 mp = _cam.ScreenToWorldPoint(new Vector3(
                Input.mousePosition.x,
                Input.mousePosition.y,
                Mathf.Abs(_cam.transform.position.z)));
            mouseWorld = new Vector2(mp.x, mp.y);
        }
        computeShader.SetFloat ("_MouseSign",  mouseSign);
        computeShader.SetFloats("_MouseWorld", mouseWorld.x, mouseWorld.y);

        int pGroups = Mathf.CeilToInt((float)numParticles / 64f);
        int kGroups = Mathf.CeilToInt((float)_paddedN     / 64f);  // ComputeKeys
        int cGroups = Mathf.CeilToInt((float)_numCells    / 64f);
        int bGroups = Mathf.CeilToInt((float)_paddedN     / 256f); // BitonicStep

        for (int s = 0; s < substeps; s++)
        {
            computeShader.SetFloat("_DeltaTime", dt);

            // 1. Reset cell ranges
            computeShader.Dispatch(_kClearCellStarts, cGroups, 1, 1);
            // 2. Gravity + predicted positions
            computeShader.Dispatch(_kGravityPredict,  pGroups, 1, 1);
            // 3. Assign cell key to each particle (fills _SortKeys/_SortIndices)
            computeShader.Dispatch(_kComputeKeys,     kGroups, 1, 1);
            // 4. Sort particles by cell key
            if (bitonicShader != null)
            {
                for (int k = 2; k <= _paddedN; k <<= 1)
                    for (int j = k >> 1; j > 0; j >>= 1)
                    {
                        bitonicShader.SetInt("_StepJ", j);
                        bitonicShader.SetInt("_StepK", k);
                        bitonicShader.Dispatch(_kBitonicStep, bGroups, 1, 1);
                    }
            }
            // 5. Find each cell's run in the sorted array
            computeShader.Dispatch(_kCalcCellStarts,  pGroups, 1, 1);
            // 6. SPH density → force → integrate
            computeShader.Dispatch(_kComputeDensity,  pGroups, 1, 1);
            computeShader.Dispatch(_kComputeForce,    pGroups, 1, 1);
            computeShader.Dispatch(_kIntegrate,       pGroups, 1, 1);
        }

        // Update render buffer once after all substeps
        computeShader.Dispatch(_kBuildRenderData, pGroups, 1, 1);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    void ReleaseBuffers()
    {
        _positions?.Release();     _positions     = null;
        _velocities?.Release();    _velocities    = null;
        _predicted?.Release();     _predicted     = null;
        _densities?.Release();     _densities     = null;
        _nearDensities?.Release(); _nearDensities = null;
        _velocityDelta?.Release(); _velocityDelta = null;
        _sortKeys?.Release();      _sortKeys      = null;
        _sortIndices?.Release();   _sortIndices   = null;
        _cellStart?.Release();     _cellStart     = null;
        _cellEnd?.Release();       _cellEnd       = null;
        _renderData?.Release();    _renderData    = null;
        _argsBuffer?.Release();    _argsBuffer    = null;
        _queryBuffer?.Release();   _queryBuffer   = null;
        _initialized = false;
    }

    void OnDestroy()
    {
        ReleaseBuffers();
        if (_glMat != null) Destroy(_glMat);
    }

    // ── GL circle overlay (Game View) ────────────────────────────────────────
    void OnRenderObject()
    {
        if (!_queryActive || !_initialized) return;
        if (Camera.current != _cam) return;     // only draw for main camera

        if (_glMat == null)
        {
            _glMat = new Material(Shader.Find("Hidden/Internal-Colored"))
                { hideFlags = HideFlags.HideAndDontSave };
        }

        _glMat.SetPass(0);
        GL.PushMatrix();
        GL.LoadIdentity();
        GL.MultMatrix(_cam.worldToCameraMatrix);    // world → view space
        GL.LoadProjectionMatrix(_cam.projectionMatrix);

        const int Segments = 64;
        float r = smoothingRadius;

        // Filled soft disc using triangle fan (semi-transparent)
        GL.Begin(GL.TRIANGLES);
        GL.Color(new Color(1f, 1f, 0f, 0.06f));
        for (int i = 0; i < Segments; i++)
        {
            float a0 = i       * Mathf.PI * 2f / Segments;
            float a1 = (i + 1) * Mathf.PI * 2f / Segments;
            GL.Vertex3(_queryWorldPos.x,                          _queryWorldPos.y,                          0f);
            GL.Vertex3(_queryWorldPos.x + Mathf.Cos(a0) * r,     _queryWorldPos.y + Mathf.Sin(a0) * r,     0f);
            GL.Vertex3(_queryWorldPos.x + Mathf.Cos(a1) * r,     _queryWorldPos.y + Mathf.Sin(a1) * r,     0f);
        }
        GL.End();

        // Crisp outline ring
        GL.Begin(GL.LINE_STRIP);
        GL.Color(new Color(1f, 0.9f, 0f, 0.9f));
        for (int i = 0; i <= Segments; i++)
        {
            float a = i * Mathf.PI * 2f / Segments;
            GL.Vertex3(_queryWorldPos.x + Mathf.Cos(a) * r,
                       _queryWorldPos.y + Mathf.Sin(a) * r, 0f);
        }
        GL.End();

        // Small crosshair at centre
        float ch = r * 0.12f;
        GL.Begin(GL.LINES);
        GL.Color(new Color(1f, 0.9f, 0f, 0.9f));
        GL.Vertex3(_queryWorldPos.x - ch, _queryWorldPos.y,       0f);
        GL.Vertex3(_queryWorldPos.x + ch, _queryWorldPos.y,       0f);
        GL.Vertex3(_queryWorldPos.x,       _queryWorldPos.y - ch, 0f);
        GL.Vertex3(_queryWorldPos.x,       _queryWorldPos.y + ch, 0f);
        GL.End();

        GL.PopMatrix();
    }

    // ── Gizmos ────────────────────────────────────────────────────────────────
    void OnDrawGizmos()
    {
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireCube(Vector3.zero, new Vector3(boundsSize.x, boundsSize.y, 0.05f));

        if (obstacleSize.x > 0f && obstacleSize.y > 0f)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireCube(
                new Vector3(obstacleCentre.x, obstacleCentre.y, 0f),
                new Vector3(obstacleSize.x, obstacleSize.y, 0.05f));
        }

        // Mouse interaction radius (scene view only)
        if (Application.isPlaying && (Input.GetMouseButton(0) || Input.GetMouseButton(1)))
        {
            Gizmos.color = Color.yellow;
            // rough circle approximation
        }
    }

    // ── Utilities ─────────────────────────────────────────────────────────────
    static int BitonicDispatchCount(int n)
    {
        int count = 0;
        for (int k = 2; k <= n; k <<= 1)
            for (int j = k >> 1; j > 0; j >>= 1)
                count++;
        return count;
    }

    static Mesh CreateQuadMesh()
    {
        var mesh = new Mesh { name = "ParticleQuad" };
        mesh.vertices = new Vector3[]
        {
            new(-0.5f, -0.5f, 0f),
            new(-0.5f,  0.5f, 0f),
            new( 0.5f,  0.5f, 0f),
            new( 0.5f, -0.5f, 0f),
        };
        mesh.uv = new Vector2[]
        {
            new(0f, 0f), new(0f, 1f), new(1f, 1f), new(1f, 0f),
        };
        mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 };
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    // ── On-screen HUD ─────────────────────────────────────────────────────────
    void OnGUI()
    {
        if (!_initialized) return;

        var style = new GUIStyle(GUI.skin.label) { fontSize = 14 };
        GUI.Label(new Rect(10, 10, 500, 100),
            $"GPU Fluid [{numParticles:N0} particles]  FPS: {1f / Time.smoothDeltaTime:F0}  " +
            $"{(_paused ? "| PAUSED" : "")}\n" +
            $"ρ₀={targetDensity:F3}  h={smoothingRadius:F4}  P={pressureMultiplier}  substeps={substeps}\n" +
            $"Space=pause  →=step  R=reset  Alt=density probe",
            style);

        // Density query tooltip near cursor
        if (_queryActive)
        {
            float density    = _queryData[0];
            float nearDensity = _queryData[1];
            float pressure   = (density - targetDensity) * pressureMultiplier;

            var screenPos = new Vector2(Input.mousePosition.x,
                                        Screen.height - Input.mousePosition.y);
            var rect = new Rect(screenPos.x + 14, screenPos.y - 10, 240, 70);

            GUI.Box(rect, "");
            GUI.Label(new Rect(rect.x + 4, rect.y + 2, rect.width - 8, rect.height - 4),
                $"Density:      {density:F3}\n" +
                $"Near Density: {nearDensity:F3}\n" +
                $"Pressure:     {pressure:F2}",
                style);
        }
    }
}
