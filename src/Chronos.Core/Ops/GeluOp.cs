using Chronos.Core.Backends;

namespace Chronos.Core.Ops;

public static class GeluOp
{
    private const float C = 0.7978845608028654f; // sqrt(2/pi)

    public static Tensor Apply(Tensor x)
    {
        var result = new Tensor(x.Shape);
        TensorBackendContext.Current.Gelu(x.Data, result.Data, x.Length);

        result.AttachGraph(new[] { x }, self =>
        {
            if (!x.RequiresGrad) return;
            var g = self.Grad!;
            var dx = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                float xv = x.Data[i];
                float inner = C * (xv + 0.044715f * xv * xv * xv);
                float t = MathF.Tanh(inner);
                float dInner = C * (1f + 3f * 0.044715f * xv * xv);
                float dTanh = (1f - t * t) * dInner;
                float derivative = 0.5f * (1f + t) + 0.5f * xv * dTanh;
                dx[i] = g[i] * derivative;
            }
            x.AccumulateGrad(dx);
        });
        return result;
    }
}
