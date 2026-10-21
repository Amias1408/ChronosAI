using Chronos.Core;
using Chronos.Core.Ops;

namespace Chronos.Model;

public sealed class Embedding : Module
{
    public readonly Tensor Weight; // [vocabSize, dim]

    public Embedding(int vocabSize, int dim, float initStd = 0.02f)
    {
        Weight = Tensor.Randn(new[] { vocabSize, dim }, initStd, requiresGrad: true);
    }

    public Tensor Forward(int[] indices, int[] indicesShape) => EmbeddingOp.Lookup(Weight, indices, indicesShape);

    public override IEnumerable<Tensor> Parameters()
    {
        yield return Weight;
    }

    public override IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "")
    {
        yield return (prefix + "weight", Weight);
    }
}
