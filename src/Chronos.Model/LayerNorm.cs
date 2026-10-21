using Chronos.Core;
using Chronos.Core.Ops;

namespace Chronos.Model;

public sealed class LayerNorm : Module
{
    public readonly Tensor Gamma;
    public readonly Tensor Beta;
    public readonly float Eps;

    public LayerNorm(int dim, float eps = 1e-5f)
    {
        Gamma = Tensor.Full(new[] { dim }, 1f, requiresGrad: true);
        Beta = Tensor.Zeros(new[] { dim }, requiresGrad: true);
        Eps = eps;
    }

    public Tensor Forward(Tensor x) => LayerNormOp.Apply(x, Gamma, Beta, Eps);

    public override IEnumerable<Tensor> Parameters()
    {
        yield return Gamma;
        yield return Beta;
    }

    public override IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "")
    {
        yield return (prefix + "gamma", Gamma);
        yield return (prefix + "beta", Beta);
    }
}
