using Chronos.Core;

namespace Chronos.Training;

public static class GradClip
{
    /// <summary>Scales every parameter's gradient down so the global L2 norm
    /// across all of them is at most maxNorm. Returns the pre-clip norm.</summary>
    public static float ClipGlobalNorm(IEnumerable<Tensor> parameters, float maxNorm)
    {
        var list = parameters as IReadOnlyList<Tensor> ?? parameters.ToList();
        double sumSq = 0;
        foreach (var p in list)
        {
            if (p.Grad is null) continue;
            foreach (var g in p.Grad) sumSq += (double)g * g;
        }
        float norm = (float)Math.Sqrt(sumSq);

        if (norm > maxNorm && norm > 0f)
        {
            float scale = maxNorm / norm;
            foreach (var p in list)
            {
                if (p.Grad is null) continue;
                for (int i = 0; i < p.Grad.Length; i++) p.Grad[i] *= scale;
            }
        }
        return norm;
    }
}
