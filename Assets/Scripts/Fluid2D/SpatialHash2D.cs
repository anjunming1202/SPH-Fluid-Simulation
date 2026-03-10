using UnityEngine;

namespace Fluid2D
{
    /// <summary>
    /// GC-free uniform-grid spatial hash using a fixed int[] array instead of Dictionary.
    /// Kept as managed C# fallback; main simulation uses Burst jobs (FluidJobs2D).
    /// </summary>
    public class SpatialHash2D
    {
        float _invCellSize;
        int   _gridW, _gridH;
        float _originX, _originY;
        int[] _gridHead;   // size = gridW * gridH, -1 = empty
        int[] _next;       // linked list within each cell

        public void Init(int capacity, float cellSize, float boundsW = 16f, float boundsH = 9f)
        {
            _invCellSize = 1f / cellSize;
            int pad = 4;
            _gridW   = Mathf.CeilToInt(boundsW / cellSize) + pad * 2;
            _gridH   = Mathf.CeilToInt(boundsH / cellSize) + pad * 2;
            _originX = -boundsW * 0.5f - pad * cellSize;
            _originY = -boundsH * 0.5f - pad * cellSize;
            _gridHead = new int[_gridW * _gridH];
            _next     = new int[capacity];
        }

        /// <summary>Rebuild grid from current positions. Call once per sub-step.</summary>
        public void Build(Vector2[] positions, int count)
        {
            System.Array.Fill(_gridHead, -1);
            for (int i = 0; i < count; i++)
            {
                int cx   = CellX(positions[i].x);
                int cy   = CellY(positions[i].y);
                int cell = cy * _gridW + cx;
                _next[i]        = _gridHead[cell];
                _gridHead[cell] = i;
            }
        }

        /// <summary>
        /// Fills <paramref name="results"/> with indices in 3x3 grid around <paramref name="pos"/>.
        /// Returns count written. Zero heap allocations.
        /// </summary>
        public int QueryInto(Vector2 pos, int[] results)
        {
            int count = 0;
            int cx = CellX(pos.x), cy = CellY(pos.y);
            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                int nx = cx + dx, ny = cy + dy;
                if ((uint)nx >= (uint)_gridW || (uint)ny >= (uint)_gridH) continue;
                int idx = _gridHead[ny * _gridW + nx];
                while (idx >= 0)
                {
                    if (count < results.Length) results[count++] = idx;
                    idx = _next[idx];
                }
            }
            return count;
        }

        int CellX(float x) => Mathf.Clamp(Mathf.FloorToInt((x - _originX) * _invCellSize), 0, _gridW - 1);
        int CellY(float y) => Mathf.Clamp(Mathf.FloorToInt((y - _originY) * _invCellSize), 0, _gridH - 1);
    }
}
