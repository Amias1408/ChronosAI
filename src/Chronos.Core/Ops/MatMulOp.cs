using Chronos.Core.Backends;

namespace Chronos.Core.Ops;

public static class MatMulOp
{
    /// <summary>x: [...,K] (any leading/batch shape), weight: [K,N] (rank 2, no batch).
    /// Flattens x's leading dims into one big matmul — this is what a Linear layer needs.</summary>
    public static Tensor Linear(Tensor x, Tensor weight)
    {
        if (weight.Rank != 2)
            throw new InvalidOperationException($"Linear: weight must be rank 2, got [{string.Join(',', weight.Shape)}].");
        int k = x.Shape[^1];
        if (weight.Shape[0] != k)
            throw new InvalidOperationException($"Linear: input last dim {k} must match weight.Shape[0]={weight.Shape[0]}.");
        int n = weight.Shape[1];
        int mPrime = x.Length / k;

        var outShape = Shapes.Concat(x.Shape[..^1], n);
        var result = new Tensor(outShape);
        TensorBackendContext.Current.MatMul(x.Data, weight.Data, result.Data, 1, mPrime, k, n);

        result.AttachGraph(new[] { x, weight }, self =>
        {
            var g = self.Grad!;
            if (x.RequiresGrad)
            {
                var wT = Transpose2D(weight.Data, k, n);
                var dx = new float[x.Length];
                TensorBackendContext.Current.MatMul(g, wT, dx, 1, mPrime, n, k);
                x.AccumulateGrad(dx);
            }
            if (weight.RequiresGrad)
            {
                var xT = Transpose2D(x.Data, mPrime, k);
                var dw = new float[k * n];
                TensorBackendContext.Current.MatMul(xT, g, dw, 1, k, mPrime, n);
                weight.AccumulateGrad(dw);
            }
        });
        return result;
    }

    /// <summary>Like Linear, but weight is stored as [N,K] (e.g. a weight-tied
    /// embedding table [vocab,dim]) and used transposed: y = x @ weight^T. Both
    /// the embedding-lookup use and this use accumulate into the same Tensor's
    /// Grad, so weight tying falls out of ordinary reference identity.</summary>
    public static Tensor LinearTiedTransposed(Tensor x, Tensor weightNK)
    {
        int k = x.Shape[^1];
        int n = weightNK.Shape[0];
        if (weightNK.Shape[1] != k)
            throw new InvalidOperationException(
                $"LinearTiedTransposed: weight.Shape[1]={weightNK.Shape[1]} must match input last dim {k}.");
        int mPrime = x.Length / k;

        var wT = Transpose2D(weightNK.Data, n, k); // -> [k,n]
        var outShape = Shapes.Concat(x.Shape[..^1], n);
        var result = new Tensor(outShape);
        TensorBackendContext.Current.MatMul(x.Data, wT, result.Data, 1, mPrime, k, n);

        result.AttachGraph(new[] { x, weightNK }, self =>
        {
            var g = self.Grad!;
            if (x.RequiresGrad)
            {
                var dx = new float[x.Length];
                TensorBackendContext.Current.MatMul(g, weightNK.Data, dx, 1, mPrime, n, k);
                x.AccumulateGrad(dx);
            }
            if (weightNK.RequiresGrad)
            {
                var gT = Transpose2D(g, mPrime, n); // [n, mPrime]
                var dw = new float[n * k];
                TensorBackendContext.Current.MatMul(gT, x.Data, dw, 1, n, mPrime, k);
                weightNK.AccumulateGrad(dw);
            }
        });
        return result;
    }

    /// <summary>a: [...,M,K], b: [...,K,N] with identical leading (batch) dims —
    /// used for Q@K^T and Attn@V in multi-head attention.</summary>
    public static Tensor Batched(Tensor a, Tensor b)
    {
        int rank = a.Rank;
        if (b.Rank != rank)
            throw new InvalidOperationException("Batched: a and b must have the same rank.");
        int m = a.Shape[^2], k = a.Shape[^1];
        int k2 = b.Shape[^2], n = b.Shape[^1];
        if (k != k2)
            throw new InvalidOperationException($"Batched: inner dims mismatch ({k} vs {k2}).");
        for (int i = 0; i < rank - 2; i++)
            if (a.Shape[i] != b.Shape[i])
                throw new InvalidOperationException("Batched: batch dims must match exactly.");

        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= a.Shape[i];

        var outShape = (int[])a.Shape.Clone();
        outShape[^1] = n;
        var result = new Tensor(outShape);
        TensorBackendContext.Current.MatMul(a.Data, b.Data, result.Data, batch, m, k, n);

        result.AttachGraph(new[] { a, b }, self =>
        {
            var g = self.Grad!;
            if (a.RequiresGrad)
            {
                var bT = TransposeBatchedLast2(b.Data, batch, k, n);
                var da = new float[a.Length];
                TensorBackendContext.Current.MatMul(g, bT, da, batch, m, n, k);
                a.AccumulateGrad(da);
            }
            if (b.RequiresGrad)
            {
                var aT = TransposeBatchedLast2(a.Data, batch, m, k);
                var db = new float[b.Length];
                TensorBackendContext.Current.MatMul(aT, g, db, batch, k, m, n);
                b.AccumulateGrad(db);
            }
        });
        return result;
    }

    private static float[] Transpose2D(float[] data, int rows, int cols)
    {
        var result = new float[data.Length];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j * rows + i] = data[i * cols + j];
        return result;
    }

    private static float[] TransposeBatchedLast2(float[] data, int batch, int rows, int cols)
    {
        var result = new float[data.Length];
        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            int baseOff = bIdx * rows * cols;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[baseOff + j * rows + i] = data[baseOff + i * cols + j];
        }
        return result;
    }
}
