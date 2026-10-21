using Chronos.Core;
using Chronos.Core.Ops;

namespace Chronos.Model;

public sealed class MultiHeadSelfAttention : Module
{
    private readonly int _nEmbd;
    private readonly int _nHead;
    private readonly int _headDim;
    private readonly int _blockSize;
    private readonly float _dropoutP;
    private readonly Tensor _causalMask; // [blockSize,blockSize], RequiresGrad=false — not a trainable parameter

    public readonly Linear QkvProj;
    public readonly Linear OutProj;

    public MultiHeadSelfAttention(int nEmbd, int nHead, int blockSize, float dropoutP, int nLayer)
    {
        if (nEmbd % nHead != 0)
            throw new ArgumentException($"nEmbd ({nEmbd}) must be divisible by nHead ({nHead}).");

        _nEmbd = nEmbd;
        _nHead = nHead;
        _headDim = nEmbd / nHead;
        _blockSize = blockSize;
        _dropoutP = dropoutP;

        QkvProj = new Linear(nEmbd, 3 * nEmbd);
        OutProj = new Linear(nEmbd, nEmbd);
        OutProj.ScaleResidualInit(nLayer);

        _causalMask = BuildCausalMask(blockSize);
    }

    private static Tensor BuildCausalMask(int t)
    {
        var mask = new Tensor(new[] { t, t });
        for (int i = 0; i < t; i++)
            for (int j = 0; j < t; j++)
                mask.Data[i * t + j] = j <= i ? 0f : float.NegativeInfinity;
        return mask;
    }

    public Tensor Forward(Tensor x, bool training)
    {
        int b = x.Shape[0], t = x.Shape[1];

        var qkv = QkvProj.Forward(x); // [B,T,3*nEmbd]
        var parts = ShapeOps.Split(qkv, new[] { _nEmbd, _nEmbd, _nEmbd });
        var q = ReshapeHeads(parts[0], b, t);
        var k = ReshapeHeads(parts[1], b, t);
        var v = ReshapeHeads(parts[2], b, t); // each: [B,nHead,T,headDim]

        var kT = ShapeOps.Transpose(k, 2, 3); // [B,nHead,headDim,T]
        var scores = MatMulOp.Batched(q, kT); // [B,nHead,T,T]
        scores = Elementwise.Scale(scores, 1f / MathF.Sqrt(_headDim));
        scores = Elementwise.Add(scores, t == _blockSize ? _causalMask : BuildCausalMask(t));

        var weights = SoftmaxOp.Apply(scores);
        weights = DropoutOp.Apply(weights, _dropoutP, training);

        var attnOut = MatMulOp.Batched(weights, v); // [B,nHead,T,headDim]
        var merged = ShapeOps.Transpose(attnOut, 1, 2); // [B,T,nHead,headDim]
        merged = ShapeOps.Reshape(merged, new[] { b, t, _nEmbd });

        var output = OutProj.Forward(merged);
        return DropoutOp.Apply(output, _dropoutP, training);
    }

    private Tensor ReshapeHeads(Tensor x, int b, int t)
    {
        var reshaped = ShapeOps.Reshape(x, new[] { b, t, _nHead, _headDim });
        return ShapeOps.Transpose(reshaped, 1, 2); // [B,nHead,T,headDim]
    }

    public override IEnumerable<Tensor> Parameters()
    {
        foreach (var p in QkvProj.Parameters()) yield return p;
        foreach (var p in OutProj.Parameters()) yield return p;
    }

    public override IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "")
    {
        foreach (var np in QkvProj.NamedParameters(prefix + "qkv_proj.")) yield return np;
        foreach (var np in OutProj.NamedParameters(prefix + "out_proj.")) yield return np;
    }
}
