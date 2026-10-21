using Chronos.Core.Backends;

namespace Chronos.Core.Ops;

public static class SoftmaxOp
{
    /// <summary>Softmax over the last axis.</summary>
    public static Tensor Apply(Tensor x)
    {
        int cols = x.Shape[^1];
        int rows = x.Length / cols;
        var result = new Tensor(x.Shape);
        TensorBackendContext.Current.Softmax(x.Data, result.Data, rows, cols);

        result.AttachGraph(new[] { x }, self =>
        {
            if (!x.RequiresGrad) return;
            var g = self.Grad!;
            var y = self.Data;
            var dx = new float[x.Length];
            for (int r = 0; r < rows; r++)
            {
                int off = r * cols;
                float dot = 0f;
                for (int j = 0; j < cols; j++) dot += g[off + j] * y[off + j];
                for (int j = 0; j < cols; j++) dx[off + j] = y[off + j] * (g[off + j] - dot);
            }
            x.AccumulateGrad(dx);
        });
        return result;
    }
}
