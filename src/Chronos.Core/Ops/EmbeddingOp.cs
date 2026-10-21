namespace Chronos.Core.Ops;

public static class EmbeddingOp
{
    /// <summary>table: [V,D]. indices is a flat row-major buffer shaped by
    /// indicesShape; output shape is indicesShape + [D].</summary>
    public static Tensor Lookup(Tensor table, int[] indices, int[] indicesShape)
    {
        int d = table.Shape[^1];
        if (Shapes.Numel(indicesShape) != indices.Length)
            throw new InvalidOperationException("Embedding: indices length does not match indicesShape.");

        var outShape = Shapes.Concat(indicesShape, d);
        var result = new Tensor(outShape);
        for (int i = 0; i < indices.Length; i++)
            Array.Copy(table.Data, indices[i] * d, result.Data, i * d, d);

        result.AttachGraph(new[] { table }, self =>
        {
            if (!table.RequiresGrad) return;
            table.EnsureGrad();
            var g = self.Grad!;
            var tg = table.Grad!;
            for (int i = 0; i < indices.Length; i++)
            {
                int rOff = indices[i] * d, gOff = i * d;
                for (int j = 0; j < d; j++) tg[rOff + j] += g[gOff + j];
            }
        });
        return result;
    }
}
