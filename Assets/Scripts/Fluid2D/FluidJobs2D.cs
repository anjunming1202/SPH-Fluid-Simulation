using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Fluid2D
{
    // ── Job 1: Apply gravity + predict next positions ──────────────────────────
    [BurstCompile]
    public struct GravityPredictJob : IJobParallelFor
    {
        public float dt;
        public float gravity;
        public NativeArray<float2> velocities;
        [ReadOnly] public NativeArray<float2> positions;
        [WriteOnly] public NativeArray<float2> predicted;

        public void Execute(int i)
        {
            float2 v = velocities[i];
            v.y += gravity * dt;
            velocities[i] = v;
            predicted[i]  = positions[i] + v * dt;
        }
    }

    // ── Job 2: Build fixed-array uniform grid (serial — avoids atomic issues) ──
    [BurstCompile]
    public struct BuildGridJob : IJob
    {
        [ReadOnly] public NativeArray<float2> predicted;
        public NativeArray<int> gridHead;
        public NativeArray<int> gridNext;
        public int count;
        public int gridW;
        public int gridH;
        public float2 gridOrigin;
        public float  invCellSize;

        public void Execute()
        {
            for (int i = 0; i < gridHead.Length; i++) gridHead[i] = -1;
            for (int i = 0; i < count; i++)
            {
                int cx   = math.clamp((int)math.floor((predicted[i].x - gridOrigin.x) * invCellSize), 0, gridW - 1);
                int cy   = math.clamp((int)math.floor((predicted[i].y - gridOrigin.y) * invCellSize), 0, gridH - 1);
                int cell = cy * gridW + cx;
                gridNext[i]    = gridHead[cell];
                gridHead[cell] = i;
            }
        }
    }

    // ── Job 3: Compute density and near-density ────────────────────────────────
    [BurstCompile]
    public struct DensityJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> predicted;
        [WriteOnly] public NativeArray<float> densities;
        [WriteOnly] public NativeArray<float> nearDensities;
        [ReadOnly] public NativeArray<int> gridHead;
        [ReadOnly] public NativeArray<int> gridNext;
        public int   gridW;
        public int   gridH;
        public float2 gridOrigin;
        public float  invCellSize;
        public float  h;

        public void Execute(int i)
        {
            float2 pi = predicted[i];
            int cx = math.clamp((int)math.floor((pi.x - gridOrigin.x) * invCellSize), 0, gridW - 1);
            int cy = math.clamp((int)math.floor((pi.y - gridOrigin.y) * invCellSize), 0, gridH - 1);

            float d = 0f, nd = 0f;
            float h2 = h * h;

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                int nx = cx + dx, ny = cy + dy;
                if ((uint)nx >= (uint)gridW || (uint)ny >= (uint)gridH) continue;
                int j = gridHead[ny * gridW + nx];
                while (j >= 0)
                {
                    float2 diff = pi - predicted[j];
                    float r2 = math.dot(diff, diff);
                    if (r2 < h2)
                    {
                        float r = math.sqrt(r2);
                        float q = 1f - r / h;
                        d  += q * q;
                        nd += q * q * q;
                    }
                    j = gridNext[j];
                }
            }

            densities[i]     = d;
            nearDensities[i] = nd;
        }
    }

    // ── Job 4: Compute pressure + viscosity velocity delta ─────────────────────
    [BurstCompile]
    public struct ForceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> predicted;
        [ReadOnly] public NativeArray<float>  densities;
        [ReadOnly] public NativeArray<float>  nearDensities;
        [ReadOnly] public NativeArray<float2> velocities;
        [WriteOnly] public NativeArray<float2> velocityDelta;
        [ReadOnly] public NativeArray<int> gridHead;
        [ReadOnly] public NativeArray<int> gridNext;
        public int   gridW;
        public int   gridH;
        public float2 gridOrigin;
        public float  invCellSize;
        public float  h;
        public float  targetDensity;
        public float  pressureMultiplier;
        public float  nearPressureMultiplier;
        public float  viscosityStrength;
        public float  dt;

        public void Execute(int i)
        {
            float2 pi  = predicted[i];
            int cx = math.clamp((int)math.floor((pi.x - gridOrigin.x) * invCellSize), 0, gridW - 1);
            int cy = math.clamp((int)math.floor((pi.y - gridOrigin.y) * invCellSize), 0, gridH - 1);

            float p_i  = (densities[i]     - targetDensity) * pressureMultiplier;
            float np_i =  nearDensities[i] * nearPressureMultiplier;
            float2 vi  = velocities[i];
            float2 force = new float2(0f, 0f);
            const float eps = 1e-5f;
            float h2 = h * h;

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                int nx = cx + dx, ny = cy + dy;
                if ((uint)nx >= (uint)gridW || (uint)ny >= (uint)gridH) continue;
                int j = gridHead[ny * gridW + nx];
                while (j >= 0)
                {
                    if (j != i)
                    {
                        float2 diff = pi - predicted[j];
                        float r2 = math.dot(diff, diff);
                        if (r2 < h2)
                        {
                            float r = math.sqrt(r2);
                            float q = 1f - r / h;
                            float2 dir = r > eps ? diff / r : new float2(0f, 1f);

                            float p_j  = (densities[j]     - targetDensity) * pressureMultiplier;
                            float np_j =  nearDensities[j] * nearPressureMultiplier;
                            float sharedP  = (p_i + p_j)  * 0.5f;
                            float sharedNP = (np_i + np_j) * 0.5f;

                            // Pressure force (gradient of q^2 and q^3 kernels)
                            force += dir * (sharedP  * 2f * q     / h
                                         + sharedNP * 3f * q * q / h);
                            // Viscosity: damp relative velocity
                            force += (velocities[j] - vi) * (q * viscosityStrength);
                        }
                    }
                    j = gridNext[j];
                }
            }

            float rho = math.max(densities[i], eps);
            velocityDelta[i] = force / rho * dt;
        }
    }

    // ── Job 5: Integrate + resolve collisions ─────────────────────────────────
    [BurstCompile]
    public struct IntegrateJob : IJobParallelFor
    {
        public NativeArray<float2> velocities;
        public NativeArray<float2> positions;
        [ReadOnly] public NativeArray<float2> velocityDelta;
        public float  dt;
        public float2 boundsHalf;
        public float  collisionDamping;
        public float  smoothingRadius;
        public float2 obstacleCentre;
        public float2 obstacleHalf;
        public bool   hasObstacle;

        public void Execute(int i)
        {
            float2 v = velocities[i] + velocityDelta[i];
            float2 p = positions[i]  + v * dt;
            float  r = smoothingRadius;

            // World bounds
            if (p.x < -boundsHalf.x + r) { p.x = -boundsHalf.x + r; v.x =  math.abs(v.x) * collisionDamping; }
            if (p.x >  boundsHalf.x - r) { p.x =  boundsHalf.x - r; v.x = -math.abs(v.x) * collisionDamping; }
            if (p.y < -boundsHalf.y + r) { p.y = -boundsHalf.y + r; v.y =  math.abs(v.y) * collisionDamping; }
            if (p.y >  boundsHalf.y - r) { p.y =  boundsHalf.y - r; v.y = -math.abs(v.y) * collisionDamping; }

            // Obstacle AABB
            if (hasObstacle)
            {
                float2 lo = obstacleCentre - obstacleHalf;
                float2 hi = obstacleCentre + obstacleHalf;
                if (p.x > lo.x - r && p.x < hi.x + r && p.y > lo.y - r && p.y < hi.y + r)
                {
                    float dL = math.abs(p.x - lo.x);
                    float dR = math.abs(p.x - hi.x);
                    float dD = math.abs(p.y - lo.y);
                    float dU = math.abs(p.y - hi.y);
                    float minD = math.min(math.min(dL, dR), math.min(dD, dU));
                    if      (minD == dL) { p.x = lo.x - r; v.x = -math.abs(v.x) * collisionDamping; }
                    else if (minD == dR) { p.x = hi.x + r; v.x =  math.abs(v.x) * collisionDamping; }
                    else if (minD == dD) { p.y = lo.y - r; v.y = -math.abs(v.y) * collisionDamping; }
                    else                 { p.y = hi.y + r; v.y =  math.abs(v.y) * collisionDamping; }
                }
            }

            velocities[i] = v;
            positions[i]  = p;
        }
    }
}
