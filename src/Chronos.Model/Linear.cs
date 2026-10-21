using Chronos.Core;
using Chronos.Core.Ops;

namespace Chronos.Model;

public sealed class Linear : Module
{
    public readonly Tensor Weight; // [inDim, outDim]
    public readonly Tensor? Bias;

    public Linear(int inDim, int outDim, bool bias = true, float initStd = 0.02f)
    {
        Weight = Tensor.Randn(new[] { inDim, outDim }, initStd, requiresGrad: true);
        Bias = bias ? Tensor.Zeros(new[] { outDim }, requiresGrad: true) : null;
    }

    public Tensor Forward(Tensor x)
    {
        var y = MatMulOp.Linear(x, Weight);
        return Bias is null ? y : Elementwise.Add(y, Bias);
    }

    /// <summary>GPT-2's residual-stability trick: shrink an output-projection
    /// layer's weight by 1/sqrt(2*nLayer) right after init.</summary>
    public void ScaleResidualInit(int nLayer)
    {
        float scale = 1f / MathF.Sqrt(2f * nLayer);
        for (int i = 0; i < Weight.Data.Length; i++) Weight.Data[i] *= scale;
    }

    public override IEnumerable<Tensor> Parameters()
    {
        yield return Weight;
        if (Bias is not null) yield return Bias;
    }

    public override IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "")
    {
        yield return (prefix + "weight", Weight);
        if (Bias is not null) yield return (prefix + "bias", Bias);
    }
}
