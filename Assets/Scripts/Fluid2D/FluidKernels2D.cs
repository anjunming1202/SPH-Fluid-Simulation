namespace Fluid2D
{
    /// <summary>
    /// Pure-static SPH kernel functions for the 2D near-density/near-pressure model.
    /// Kernel basis: q = clamp(1 - r/h, 0, 1)
    ///   density     += q^2
    ///   nearDensity += q^3
    /// </summary>
    public static class FluidKernels2D
    {
        public static float Density(float r, float h)
        {
            if (r >= h) return 0f;
            float q = 1f - r / h;
            return q * q;
        }

        public static float NearDensity(float r, float h)
        {
            if (r >= h) return 0f;
            float q = 1f - r / h;
            return q * q * q;
        }

        /// <summary>Returns |grad W|; multiply by unit direction vector in caller.</summary>
        public static float DensityGrad(float r, float h)
        {
            if (r >= h) return 0f;
            float q = 1f - r / h;
            return -2f * q / h;
        }

        public static float NearDensityGrad(float r, float h)
        {
            if (r >= h) return 0f;
            float q = 1f - r / h;
            return -3f * q * q / h;
        }
    }
}
