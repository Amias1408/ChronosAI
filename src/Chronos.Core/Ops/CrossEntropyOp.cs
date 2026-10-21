using Chronos.Core.Backends;

namespace Chronos.Core.Ops;

public static class CrossEntropyOp
{
    /// <summary>Fused softmax + NLL. logits: [...,vocab] flattened to [rows,vocab];
    /// targets has one class id per row. Returns the mean loss over rows.</summary>
    public static Tensor Apply(Tensor logits, int[] targets)
    {
        int cols = logits.Shape[^1];
        int rows = logits.Length / cols;
        if (targets.Length != rows)
            throw new InvalidOperationException("CrossEntropy: targets length must match row count.");

        var probs = new float[logits.Length];
        TensorBackendContext.Current.Softmax(logits.Data, probs, rows, cols);

        float lossSum = 0f;
        for (int r = 0; r < rows; r++)
            lossSum += -MathF.Log(MathF.Max(probs[r * cols + targets[r]], 1e-12f));

        var result = new Tensor(new[] { 1 });
        result.Data[0] = lossSum / rows;

        result.AttachGraph(new[] { logits }, self =>
        {
            if (!logits.RequiresGrad) return;
            float scale = self.Grad![0] / rows;
            var dx = new float[logits.Length];
            for (int r = 0; r < rows; r++)
            {
                int off = r * cols;
                for (int j = 0; j < cols; j++) dx[off + j] = probs[off + j] * scale;
                dx[off + targets[r]] -= scale;
            }
            logits.AccumulateGrad(dx);
        });
        return result;
    }
}
