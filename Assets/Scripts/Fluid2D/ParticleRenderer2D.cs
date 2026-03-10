using UnityEngine;
using Unity.Mathematics;

namespace Fluid2D
{
    [RequireComponent(typeof(FluidSim2D))]
    public class ParticleRenderer2D : MonoBehaviour
    {
        public Material particleMaterial;
        public float particleRadius = 0.08f;

        [Header("Debug coloring")]
        public bool colorBySpeed = true;
        public float maxSpeed = 10f;

        FluidSim2D _sim;
        Mesh _quadMesh;

        const int BatchSize = 1023;
        Matrix4x4[] _matrices = new Matrix4x4[BatchSize];
        Vector4[]   _colors   = new Vector4[BatchSize];
        MaterialPropertyBlock _mpb;

        void Awake()
        {
            _sim      = GetComponent<FluidSim2D>();
            _quadMesh = BuildQuad();
            _mpb      = new MaterialPropertyBlock();
        }

        void LateUpdate()
        {
            if (!_sim.positionsNative.IsCreated || particleMaterial == null) return;
            DrawParticles();
        }

        void DrawParticles()
        {
            int count      = _sim.particleCount;
            int batches    = Mathf.CeilToInt((float)count / BatchSize);
            float scale    = particleRadius * 2f;
            float maxSpSq  = maxSpeed * maxSpeed;

            var positions  = _sim.positionsNative;
            var velocities = _sim.velocitiesNative;

            for (int b = 0; b < batches; b++)
            {
                int start      = b * BatchSize;
                int batchCount = Mathf.Min(BatchSize, count - start);

                for (int i = 0; i < batchCount; i++)
                {
                    var pos = positions[start + i];
                    _matrices[i] = Matrix4x4.TRS(
                        new Vector3(pos.x, pos.y, 0f),
                        Quaternion.identity,
                        new Vector3(scale, scale, scale));

                    if (colorBySpeed)
                    {
                        // sqrMagnitude avoids sqrt — visually equivalent for color mapping
                        var vel = velocities[start + i];
                        float t = Mathf.Clamp01(math.dot(vel, vel) / maxSpSq);
                        _colors[i] = Color.Lerp(Color.blue, Color.red, t);
                    }
                    else
                    {
                        _colors[i] = Color.cyan;
                    }
                }

                _mpb.SetVectorArray("_Color", _colors);
                Graphics.DrawMeshInstanced(
                    _quadMesh, 0, particleMaterial, _matrices, batchCount, _mpb);
            }
        }

        static Mesh BuildQuad()
        {
            var m = new Mesh { name = "ParticleQuad" };
            m.vertices  = new Vector3[]
            {
                new Vector3(-0.5f, -0.5f, 0),
                new Vector3( 0.5f, -0.5f, 0),
                new Vector3( 0.5f,  0.5f, 0),
                new Vector3(-0.5f,  0.5f, 0)
            };
            m.uv        = new Vector2[] { new Vector2(0,0), new Vector2(1,0), new Vector2(1,1), new Vector2(0,1) };
            m.triangles = new int[] { 0, 2, 1, 0, 3, 2 };
            m.RecalculateNormals();
            return m;
        }
    }
}
