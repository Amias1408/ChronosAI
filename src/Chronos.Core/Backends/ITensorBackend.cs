namespace Chronos.Core.Backends;

/// <summary>The only place raw math loops live. Every autograd op in
/// Chronos.Core.Ops calls into the ambient <see cref="TensorBackendContext.Current"/>
/// instead of computing directly, so a future backend (e.g. GPU) is an additive
/// implementation of this interface, not a rewrite of the op graph.</summary>
public interface ITensorBackend
{
    string Name { get; }

    /// <summary>Batched matmul: a is [batch,m,k], b is [batch,k,n], dest is [batch,m,n].</summary>
    void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dest, int batch, int m, int k, int n);

    void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dest, int length);
    void Mul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dest, int length);

    /// <summary>Row-wise softmax over the last axis: src/dest are [rows,cols].</summary>
    void Softmax(ReadOnlySpan<float> src, Span<float> dest, int rows, int cols);

    /// <summary>Row-wise layer norm: x/dest are [rows,cols], gamma/beta are [cols].
    /// meanOut/rstdOut (length rows) are written so the backward pass can reuse them.</summary>
    void LayerNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta,
                   Span<float> dest, Span<float> meanOut, Span<float> rstdOut, int rows, int cols, float eps);

    void Gelu(ReadOnlySpan<float> src, Span<float> dest, int length);

    /// <summary>Row-wise sum over the last axis: src is [rows,cols], dest is [rows].</summary>
    void Sum(ReadOnlySpan<float> src, Span<float> dest, int rows, int cols);
}
