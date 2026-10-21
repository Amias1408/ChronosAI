using System.Numerics;

namespace Chronos.Core.Backends;

/// <summary>Pure-BCL CPU backend: <see cref="Vector{T}"/> SIMD (portable width —
/// SSE/AVX2/AVX-512 on x86, NEON on ARM64) plus <see cref="Parallel"/> across rows.
/// No external BLAS, no unsafe code.</summary>
public sealed class CpuTensorBackend : ITensorBackend
{
    public string Name => "cpu";

    private const int ParallelRowThreshold = 32;

    public void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dest, int batch, int m, int k, int n)
    {
        var aArr = a.ToArray();
        var bArr = b.ToArray();
        var destArr = new float[dest.Length];
        int totalRows = batch * m;
        int vw = Vector<float>.Count;

        void Row(int row)
        {
            int bi = row / m;
            int i = row % m;
            int aOff = bi * m * k + i * k;
            int bBase = bi * k * n;
            int cOff = bi * m * n + i * n;
            var cRow = destArr.AsSpan(cOff, n);
            for (int kk = 0; kk < k; kk++)
            {
                float aik = aArr[aOff + kk];
                if (aik == 0f) continue;
                var bRow = bArr.AsSpan(bBase + kk * n, n);
                var aVec = new Vector<float>(aik);
                int j = 0;
                for (; j <= n - vw; j += vw)
                    (new Vector<float>(cRow.Slice(j)) + aVec * new Vector<float>(bRow.Slice(j))).CopyTo(cRow.Slice(j));
                for (; j < n; j++) cRow[j] += aik * bRow[j];
            }
        }

        if (totalRows >= ParallelRowThreshold)
            Parallel.For(0, totalRows, Row);
        else
            for (int r = 0; r < totalRows; r++) Row(r);

        destArr.CopyTo(dest);
    }

    public void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dest, int length)
    {
        int vw = Vector<float>.Count;
        int i = 0;
        for (; i <= length - vw; i += vw)
            (new Vector<float>(a.Slice(i)) + new Vector<float>(b.Slice(i))).CopyTo(dest.Slice(i));
        for (; i < length; i++) dest[i] = a[i] + b[i];
    }

    public void Mul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dest, int length)
    {
        int vw = Vector<float>.Count;
        int i = 0;
        for (; i <= length - vw; i += vw)
            (new Vector<float>(a.Slice(i)) * new Vector<float>(b.Slice(i))).CopyTo(dest.Slice(i));
        for (; i < length; i++) dest[i] = a[i] * b[i];
    }

    public void Softmax(ReadOnlySpan<float> src, Span<float> dest, int rows, int cols)
    {
        var srcArr = src.ToArray();
        var destArr = new float[dest.Length];

        void Row(int r)
        {
            int off = r * cols;
            float max = float.NegativeInfinity;
            for (int j = 0; j < cols; j++) max = MathF.Max(max, srcArr[off + j]);
            float sum = 0f;
            for (int j = 0; j < cols; j++)
            {
                float e = MathF.Exp(srcArr[off + j] - max);
                destArr[off + j] = e;
                sum += e;
            }
            float inv = 1f / sum;
            for (int j = 0; j < cols; j++) destArr[off + j] *= inv;
        }

        if (rows >= ParallelRowThreshold) Parallel.For(0, rows, Row);
        else for (int r = 0; r < rows; r++) Row(r);

        destArr.CopyTo(dest);
    }

    public void LayerNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta,
        Span<float> dest, Span<float> meanOut, Span<float> rstdOut, int rows, int cols, float eps)
    {
        var xArr = x.ToArray();
        var gArr = gamma.ToArray();
        var bArr = beta.ToArray();
        var destArr = new float[dest.Length];
        var meanArr = new float[rows];
        var rstdArr = new float[rows];

        void Row(int r)
        {
            int off = r * cols;
            float mean = 0f;
            for (int j = 0; j < cols; j++) mean += xArr[off + j];
            mean /= cols;
            float variance = 0f;
            for (int j = 0; j < cols; j++)
            {
                float d = xArr[off + j] - mean;
                variance += d * d;
            }
            variance /= cols;
            float rstd = 1f / MathF.Sqrt(variance + eps);
            meanArr[r] = mean;
            rstdArr[r] = rstd;
            for (int j = 0; j < cols; j++)
                destArr[off + j] = (xArr[off + j] - mean) * rstd * gArr[j] + bArr[j];
        }

        if (rows >= ParallelRowThreshold) Parallel.For(0, rows, Row);
        else for (int r = 0; r < rows; r++) Row(r);

        destArr.CopyTo(dest);
        meanArr.CopyTo(meanOut);
        rstdArr.CopyTo(rstdOut);
    }

    public void Gelu(ReadOnlySpan<float> src, Span<float> dest, int length)
    {
        const float c = 0.7978845608028654f; // sqrt(2/pi)
        for (int i = 0; i < length; i++)
        {
            float xv = src[i];
            float inner = c * (xv + 0.044715f * xv * xv * xv);
            dest[i] = 0.5f * xv * (1f + MathF.Tanh(inner));
        }
    }

    public void Sum(ReadOnlySpan<float> src, Span<float> dest, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            int off = r * cols;
            float s = 0f;
            for (int j = 0; j < cols; j++) s += src[off + j];
            dest[r] = s;
        }
    }
}
