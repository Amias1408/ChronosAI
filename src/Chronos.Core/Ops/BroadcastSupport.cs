namespace Chronos.Core.Ops;

/// <summary>Shared machinery for NumPy-style broadcasting elementwise ops:
/// expand a smaller operand up to the broadcast output shape for forward,
/// and reduce (sum) a full-size gradient back down to an operand's original
/// shape for backward. Both directions reuse the same per-dim stride mapping.</summary>
internal static class BroadcastSupport
{
    public static float[] ExpandToShape(float[] data, int[] shape, int[] targetShape)
    {
        if (Shapes.ShapeEquals(shape, targetShape)) return data;

        var bStrides = Shapes.BroadcastStrides(shape, targetShape);
        var tStrides = Shapes.Strides(targetShape);
        int n = Shapes.Numel(targetShape);
        var result = new float[n];
        for (int flat = 0; flat < n; flat++)
        {
            int rem = flat, srcOffset = 0;
            for (int d = 0; d < targetShape.Length; d++)
            {
                int idx = rem / tStrides[d];
                rem %= tStrides[d];
                srcOffset += idx * bStrides[d];
            }
            result[flat] = data[srcOffset];
        }
        return result;
    }

    public static float[] ReduceGradToShape(float[] gradFull, int[] fullShape, int[] shape)
    {
        if (Shapes.ShapeEquals(fullShape, shape)) return gradFull;

        var bStrides = Shapes.BroadcastStrides(shape, fullShape);
        var fStrides = Shapes.Strides(fullShape);
        var result = new float[Shapes.Numel(shape)];
        int n = gradFull.Length;
        for (int flat = 0; flat < n; flat++)
        {
            int rem = flat, srcOffset = 0;
            for (int d = 0; d < fullShape.Length; d++)
            {
                int idx = rem / fStrides[d];
                rem %= fStrides[d];
                srcOffset += idx * bStrides[d];
            }
            result[srcOffset] += gradFull[flat];
        }
        return result;
    }
}
