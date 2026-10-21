namespace Chronos.Core;

public static class Shapes
{
    public static int Numel(ReadOnlySpan<int> shape)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        return n;
    }

    public static int[] Strides(ReadOnlySpan<int> shape)
    {
        var strides = new int[shape.Length];
        int acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    public static bool ShapeEquals(ReadOnlySpan<int> a, ReadOnlySpan<int> b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>NumPy-style broadcast: shapes are aligned from the right, missing
    /// leading dims act as 1, and each aligned pair must be equal or one of them 1.</summary>
    public static int[] Broadcast(int[] a, int[] b)
    {
        int rank = Math.Max(a.Length, b.Length);
        var result = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int ai = i < rank - a.Length ? 1 : a[i - (rank - a.Length)];
            int bi = i < rank - b.Length ? 1 : b[i - (rank - b.Length)];
            if (ai != bi && ai != 1 && bi != 1)
                throw new InvalidOperationException(
                    $"Shapes [{string.Join(',', a)}] and [{string.Join(',', b)}] are not broadcast-compatible.");
            result[i] = Math.Max(ai, bi);
        }
        return result;
    }

    /// <summary>Per-dim stride to read `shape` as if it were broadcast to `targetShape`
    /// (right-aligned): a dim that is size 1 in `shape` but larger in the target gets
    /// stride 0 so every target index reads the same source element.</summary>
    public static int[] BroadcastStrides(int[] shape, int[] targetShape)
    {
        int rank = targetShape.Length;
        var own = Strides(shape);
        var result = new int[rank];
        int offset = rank - shape.Length;
        for (int i = 0; i < rank; i++)
        {
            if (i < offset) { result[i] = 0; continue; }
            int dim = shape[i - offset];
            result[i] = dim == 1 ? 0 : own[i - offset];
        }
        return result;
    }

    public static int[] Concat(int[] a, int b)
    {
        var r = new int[a.Length + 1];
        Array.Copy(a, r, a.Length);
        r[^1] = b;
        return r;
    }
}
