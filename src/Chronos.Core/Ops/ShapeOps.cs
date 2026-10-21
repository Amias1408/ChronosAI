namespace Chronos.Core.Ops;

public static class ShapeOps
{
    /// <summary>Contiguous-only reshape: shares the underlying data array (safe
    /// because no op in this codebase ever mutates a Tensor's Data in place).</summary>
    public static Tensor Reshape(Tensor x, int[] newShape)
    {
        if (Shapes.Numel(newShape) != x.Length)
            throw new InvalidOperationException(
                $"Reshape: element count mismatch ({x.Length} vs [{string.Join(',', newShape)}]).");

        var result = Tensor.FromArray(x.Data, newShape);
        result.AttachGraph(new[] { x }, self =>
        {
            if (x.RequiresGrad) x.AccumulateGrad(self.Grad!);
        });
        return result;
    }

    /// <summary>Swaps two axes, materializing a contiguous copy.</summary>
    public static Tensor Transpose(Tensor x, int dim0, int dim1)
    {
        int rank = x.Rank;
        var newShape = (int[])x.Shape.Clone();
        (newShape[dim0], newShape[dim1]) = (newShape[dim1], newShape[dim0]);

        var oldStrides = Shapes.Strides(x.Shape);
        var newStrides = Shapes.Strides(newShape);
        var result = new Tensor(newShape);
        int n = x.Length;

        var idx = new int[rank];
        for (int flat = 0; flat < n; flat++)
        {
            int rem = flat;
            for (int d = 0; d < rank; d++) { idx[d] = rem / oldStrides[d]; rem %= oldStrides[d]; }
            (idx[dim0], idx[dim1]) = (idx[dim1], idx[dim0]);
            int newFlat = 0;
            for (int d = 0; d < rank; d++) newFlat += idx[d] * newStrides[d];
            result.Data[newFlat] = x.Data[flat];
        }

        result.AttachGraph(new[] { x }, self =>
        {
            if (!x.RequiresGrad) return;
            var g = self.Grad!;
            var dx = new float[n];
            var bIdx = new int[rank];
            for (int flat = 0; flat < n; flat++)
            {
                int rem = flat;
                for (int d = 0; d < rank; d++) { bIdx[d] = rem / oldStrides[d]; rem %= oldStrides[d]; }
                (bIdx[dim0], bIdx[dim1]) = (bIdx[dim1], bIdx[dim0]);
                int newFlat = 0;
                for (int d = 0; d < rank; d++) newFlat += bIdx[d] * newStrides[d];
                dx[flat] = g[newFlat];
            }
            x.AccumulateGrad(dx);
        });
        return result;
    }

    /// <summary>Splits along the last axis into contiguous chunks of the given sizes.</summary>
    public static Tensor[] Split(Tensor x, int[] sizes)
    {
        int lastDim = x.Shape[^1];
        int total = 0;
        foreach (var s in sizes) total += s;
        if (total != lastDim)
            throw new InvalidOperationException("Split: sizes must sum to the last dimension.");

        int rows = x.Length / lastDim;
        var offsets = new int[sizes.Length];
        int acc = 0;
        for (int s = 0; s < sizes.Length; s++) { offsets[s] = acc; acc += sizes[s]; }

        var outputs = new Tensor[sizes.Length];
        for (int s = 0; s < sizes.Length; s++)
        {
            var outShape = (int[])x.Shape.Clone();
            outShape[^1] = sizes[s];
            var t = new Tensor(outShape);
            int size = sizes[s], offset = offsets[s];
            for (int r = 0; r < rows; r++)
                Array.Copy(x.Data, r * lastDim + offset, t.Data, r * size, size);
            outputs[s] = t;
        }

        for (int s = 0; s < sizes.Length; s++)
        {
            int size = sizes[s], offset = offsets[s];
            outputs[s].AttachGraph(new[] { x }, self =>
            {
                if (!x.RequiresGrad) return;
                var dx = new float[x.Length];
                for (int r = 0; r < rows; r++)
                    Array.Copy(self.Grad!, r * size, dx, r * lastDim + offset, size);
                x.AccumulateGrad(dx);
            });
        }
        return outputs;
    }

    /// <summary>Concatenates along the last axis — the exact inverse of <see cref="Split"/>.</summary>
    public static Tensor Concat(Tensor[] tensors, int axis)
    {
        int rank = tensors[0].Rank;
        if (axis != -1 && axis != rank - 1)
            throw new NotSupportedException("Concat: only the last axis is supported.");

        int lastDim = 0;
        foreach (var t in tensors) lastDim += t.Shape[^1];
        int rows = tensors[0].Length / tensors[0].Shape[^1];

        var outShape = (int[])tensors[0].Shape.Clone();
        outShape[^1] = lastDim;
        var result = new Tensor(outShape);

        var offsets = new int[tensors.Length];
        int acc = 0;
        for (int i = 0; i < tensors.Length; i++) { offsets[i] = acc; acc += tensors[i].Shape[^1]; }

        for (int i = 0; i < tensors.Length; i++)
        {
            var t = tensors[i];
            int size = t.Shape[^1], offset = offsets[i];
            for (int r = 0; r < rows; r++)
                Array.Copy(t.Data, r * size, result.Data, r * lastDim + offset, size);
        }

        result.AttachGraph(tensors, self =>
        {
            var g = self.Grad!;
            for (int i = 0; i < tensors.Length; i++)
            {
                var t = tensors[i];
                if (!t.RequiresGrad) continue;
                int size = t.Shape[^1], offset = offsets[i];
                var dt = new float[t.Length];
                for (int r = 0; r < rows; r++)
                    Array.Copy(g, r * lastDim + offset, dt, r * size, size);
                t.AccumulateGrad(dt);
            }
        });
        return result;
    }
}
