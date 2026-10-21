using Chronos.Core.Backends;
using Chronos.Core.Rng;

namespace Chronos.Core.Ops;

public static class DropoutOp
{
    /// <summary>Inverted dropout. Returns x unchanged (no new graph node) when
    /// not training or p&lt;=0 — nothing to check numerically at that point anyway.</summary>
    public static Tensor Apply(Tensor x, float p, bool training)
    {
        if (!training || p <= 0f) return x;

        float keep = 1f - p;
        float scale = 1f / keep;
        var mask = new float[x.Length];
        for (int i = 0; i < mask.Length; i++)
            mask[i] = RngProvider.NextBernoulli(keep) ? scale : 0f;

        var result = new Tensor(x.Shape);
        TensorBackendContext.Current.Mul(x.Data, mask, result.Data, x.Length);

        result.AttachGraph(new[] { x }, self =>
        {
            if (!x.RequiresGrad) return;
            var g = self.Grad!;
            var dx = new float[x.Length];
            TensorBackendContext.Current.Mul(g, mask, dx, x.Length);
            x.AccumulateGrad(dx);
        });
        return result;
    }
}
