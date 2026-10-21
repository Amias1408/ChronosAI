using Chronos.Core;

namespace Chronos.Model;

public abstract class Module
{
    public abstract IEnumerable<Tensor> Parameters();
    public abstract IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "");

    public void ZeroGrad()
    {
        foreach (var p in Parameters()) p.ZeroGrad();
    }
}
