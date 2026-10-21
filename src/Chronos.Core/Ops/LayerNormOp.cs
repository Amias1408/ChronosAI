using Chronos.Core.Backends;

namespace Chronos.Core.Ops;

public static class LayerNormOp
{
    public static Tensor Apply(Tensor x, Tensor gamma, Tensor beta, float eps = 1e-5f)
    {
        int cols = x.Shape[^1];
        int rows = x.Length / cols;
        var result = new Tensor(x.Shape);
        var mean = new float[rows];
        var rstd = new float[rows];
        TensorBackendContext.Current.LayerNorm(x.Data, gamma.Data, beta.Data, result.Data, mean, rstd, rows, cols, eps);

        result.AttachGraph(new[] { x, gamma, beta }, self =>
        {
            var g = self.Grad!;
            if (x.RequiresGrad)
            {
                var dx = new float[x.Length];
                for (int r = 0; r < rows; r++)
                {
                    int off = r * cols;
                    float rs = rstd[r], m = mean[r];
                    float sumDy = 0f, sumDyXhat = 0f;
                    for (int j = 0; j < cols; j++)
                    {
                        float xhat = (x.Data[off + j] - m) * rs;
                        float dy = g[off + j] * gamma.Data[j];
                        sumDy += dy;
                        sumDyXhat += dy * xhat;
                    }
                    for (int j = 0; j < cols; j++)
                    {
                        float xhat = (x.Data[off + j] - m) * rs;
                        float dy = g[off + j] * gamma.Data[j];
                        dx[off + j] = rs * (dy - sumDy / cols - xhat * sumDyXhat / cols);
                    }
                }
                x.AccumulateGrad(dx);
            }
            if (gamma.RequiresGrad)
            {
                var dGamma = new float[cols];
                for (int r = 0; r < rows; r++)
                {
                    int off = r * cols;
                    float rs = rstd[r], m = mean[r];
                    for (int j = 0; j < cols; j++)
                        dGamma[j] += g[off + j] * (x.Data[off + j] - m) * rs;
                }
                gamma.AccumulateGrad(dGamma);
            }
            if (beta.RequiresGrad)
            {
                var dBeta = new float[cols];
                for (int r = 0; r < rows; r++)
                {
                    int off = r * cols;
                    for (int j = 0; j < cols; j++) dBeta[j] += g[off + j];
                }
                beta.AccumulateGrad(dBeta);
            }
        });
        return result;
    }
}
