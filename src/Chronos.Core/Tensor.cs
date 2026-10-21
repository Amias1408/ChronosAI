using System.Threading;
using Chronos.Core.Rng;

namespace Chronos.Core;

/// <summary>A dense float tensor with an optional reverse-mode autograd graph.
/// Storage is always contiguous, row-major. The graph is define-by-run: every
/// op that produces a tensor records its parents and a backward closure, and
/// <see cref="Backward"/> walks that graph from a scalar loss back to the leaves.</summary>
public sealed class Tensor
{
    private static int _nextId;

    public float[] Data;
    public float[]? Grad;
    public int[] Shape;
    public bool RequiresGrad;

    internal Tensor[]? Parents;
    internal Action<Tensor>? BackwardFn;
    internal readonly int Id = Interlocked.Increment(ref _nextId);

    public int Rank => Shape.Length;
    public int Length => Data.Length;

    public Tensor(int[] shape, bool requiresGrad = false)
    {
        Shape = shape;
        Data = new float[Shapes.Numel(shape)];
        RequiresGrad = requiresGrad && !NoGradScope.IsActive;
    }

    private Tensor(float[] data, int[] shape, bool requiresGrad)
    {
        Shape = shape;
        Data = data;
        RequiresGrad = requiresGrad && !NoGradScope.IsActive;
    }

    public static Tensor FromArray(float[] data, int[] shape, bool requiresGrad = false)
    {
        if (data.Length != Shapes.Numel(shape))
            throw new ArgumentException($"Data length {data.Length} does not match shape [{string.Join(',', shape)}].");
        return new Tensor(data, shape, requiresGrad);
    }

    public static Tensor Zeros(int[] shape, bool requiresGrad = false) => new(shape, requiresGrad);

    public static Tensor Full(int[] shape, float value, bool requiresGrad = false)
    {
        var t = new Tensor(shape, requiresGrad);
        Array.Fill(t.Data, value);
        return t;
    }

    public static Tensor Randn(int[] shape, float std, bool requiresGrad = false)
    {
        var t = new Tensor(shape, requiresGrad);
        RngProvider.FillGaussian(t.Data, std);
        return t;
    }

    public void EnsureGrad() => Grad ??= new float[Data.Length];

    public void ZeroGrad()
    {
        if (Grad is null) return;
        Array.Clear(Grad);
    }

    /// <summary>Accumulates <paramref name="delta"/> (same length as Data) into Grad.</summary>
    public void AccumulateGrad(ReadOnlySpan<float> delta)
    {
        EnsureGrad();
        var g = Grad!;
        for (int i = 0; i < g.Length; i++) g[i] += delta[i];
    }

    /// <summary>Wires this tensor into the graph as the output of an op over
    /// <paramref name="parents"/>. A no-op under <see cref="NoGradScope"/>, or
    /// if none of the parents require grad.</summary>
    internal void AttachGraph(Tensor[] parents, Action<Tensor> backwardFn)
    {
        if (NoGradScope.IsActive) return;
        bool any = false;
        foreach (var p in parents)
            if (p.RequiresGrad) { any = true; break; }
        if (!any) return;

        RequiresGrad = true;
        Parents = parents;
        BackwardFn = backwardFn;
    }

    /// <summary>Runs reverse-mode autodiff from this (scalar) tensor back to every
    /// leaf that requires grad, accumulating into each leaf's <see cref="Grad"/>.</summary>
    public void Backward()
    {
        if (!RequiresGrad)
            throw new InvalidOperationException("Backward() called on a tensor that does not require grad.");
        if (Data.Length != 1)
            throw new InvalidOperationException($"Backward() requires a scalar tensor, got shape [{string.Join(',', Shape)}].");

        EnsureGrad();
        Grad![0] = 1f;

        var postorder = new List<Tensor>();
        var visited = new HashSet<int>();
        var stack = new Stack<(Tensor node, bool expanded)>();
        stack.Push((this, false));
        while (stack.Count > 0)
        {
            var (node, expanded) = stack.Pop();
            if (expanded)
            {
                postorder.Add(node);
                continue;
            }
            if (!visited.Add(node.Id)) continue;
            stack.Push((node, true));
            if (node.Parents is { } parents)
                foreach (var p in parents)
                    if (p.RequiresGrad && !visited.Contains(p.Id))
                        stack.Push((p, false));
        }

        for (int i = postorder.Count - 1; i >= 0; i--)
            postorder[i].BackwardFn?.Invoke(postorder[i]);
    }
}
