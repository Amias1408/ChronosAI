using Chronos.Core.Backends;

namespace Chronos.Core.Ops;

public static class Reduce
{
    public static Tensor Sum(Tensor a)
    {
        var result = new Tensor(new[] { 1 });
        var tmp = new float[1];
        TensorBackendContext.Current.Sum(a.Data, tmp, 1, a.Length);
        result.Data[0] = tmp[0];

        result.AttachGraph(new[] { a }, self =>
        {
            if (!a.RequiresGrad) return;
            var da = new float[a.Length];
            Array.Fill(da, self.Grad![0]);
            a.AccumulateGrad(da);
        });
        return result;
    }

    public static Tensor Mean(Tensor a) => Elementwise.Scale(Sum(a), 1f / a.Length);
}
