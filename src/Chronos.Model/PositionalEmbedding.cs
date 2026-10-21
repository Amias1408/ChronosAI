using Chronos.Core;
using Chronos.Core.Ops;

namespace Chronos.Model;

/// <summary>Learned absolute position table (RoPE deferred to a later pass).</summary>
public sealed class PositionalEmbedding : Module
{
    public readonly Tensor Weight; // [blockSize, dim]

    public PositionalEmbedding(int blockSize, int dim, float initStd = 0.02f)
    {
        Weight = Tensor.Randn(new[] { blockSize, dim }, initStd, requiresGrad: true);
    }

    public Tensor Forward(int seqLen)
    {
        var indices = new int[seqLen];
        for (int i = 0; i < seqLen; i++) indices[i] = i;
        return EmbeddingOp.Lookup(Weight, indices, new[] { seqLen });
    }

    public override IEnumerable<Tensor> Parameters()
    {
        yield return Weight;
    }

    public override IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "")
    {
        yield return (prefix + "weight", Weight);
    }
}
