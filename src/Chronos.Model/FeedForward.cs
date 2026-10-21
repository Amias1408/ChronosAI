using Chronos.Core;
using Chronos.Core.Ops;

namespace Chronos.Model;

public sealed class FeedForward : Module
{
    private readonly float _dropoutP;
    public readonly Linear Fc1;
    public readonly Linear Fc2;

    public FeedForward(int nEmbd, int mult, float dropoutP, int nLayer)
    {
        Fc1 = new Linear(nEmbd, mult * nEmbd);
        Fc2 = new Linear(mult * nEmbd, nEmbd);
        Fc2.ScaleResidualInit(nLayer);
        _dropoutP = dropoutP;
    }

    public Tensor Forward(Tensor x, bool training)
    {
        var h = GeluOp.Apply(Fc1.Forward(x));
        var y = Fc2.Forward(h);
        return DropoutOp.Apply(y, _dropoutP, training);
    }

    public override IEnumerable<Tensor> Parameters()
    {
        foreach (var p in Fc1.Parameters()) yield return p;
        foreach (var p in Fc2.Parameters()) yield return p;
    }

    public override IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "")
    {
        foreach (var np in Fc1.NamedParameters(prefix + "fc1.")) yield return np;
        foreach (var np in Fc2.NamedParameters(prefix + "fc2.")) yield return np;
    }
}
