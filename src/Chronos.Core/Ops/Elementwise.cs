using Chronos.Core.Backends;

namespace Chronos.Core.Ops;

public static class Elementwise
{
    public static Tensor Add(Tensor a, Tensor b)
    {
        var outShape = Shapes.Broadcast(a.Shape, b.Shape);
        var aExp = BroadcastSupport.ExpandToShape(a.Data, a.Shape, outShape);
        var bExp = BroadcastSupport.ExpandToShape(b.Data, b.Shape, outShape);
        var result = new Tensor(outShape);
        TensorBackendContext.Current.Add(aExp, bExp, result.Data, result.Length);

        result.AttachGraph(new[] { a, b }, self =>
        {
            var g = self.Grad!;
            if (a.RequiresGrad) a.AccumulateGrad(BroadcastSupport.ReduceGradToShape(g, outShape, a.Shape));
            if (b.RequiresGrad) b.AccumulateGrad(BroadcastSupport.ReduceGradToShape(g, outShape, b.Shape));
        });
        return result;
    }

    public static Tensor Mul(Tensor a, Tensor b)
    {
        var outShape = Shapes.Broadcast(a.Shape, b.Shape);
        var aExp = BroadcastSupport.ExpandToShape(a.Data, a.Shape, outShape);
        var bExp = BroadcastSupport.ExpandToShape(b.Data, b.Shape, outShape);
        var result = new Tensor(outShape);
        TensorBackendContext.Current.Mul(aExp, bExp, result.Data, result.Length);

        result.AttachGraph(new[] { a, b }, self =>
        {
            var g = self.Grad!;
            if (a.RequiresGrad)
            {
                var da = new float[g.Length];
                TensorBackendContext.Current.Mul(g, bExp, da, g.Length);
                a.AccumulateGrad(BroadcastSupport.ReduceGradToShape(da, outShape, a.Shape));
            }
            if (b.RequiresGrad)
            {
                var db = new float[g.Length];
                TensorBackendContext.Current.Mul(g, aExp, db, g.Length);
                b.AccumulateGrad(BroadcastSupport.ReduceGradToShape(db, outShape, b.Shape));
            }
        });
        return result;
    }

    public static Tensor Scale(Tensor a, float scalar)
    {
        var result = new Tensor(a.Shape);
        for (int i = 0; i < a.Length; i++) result.Data[i] = a.Data[i] * scalar;

        result.AttachGraph(new[] { a }, self =>
        {
            if (!a.RequiresGrad) return;
            var g = self.Grad!;
            var da = new float[g.Length];
            for (int i = 0; i < da.Length; i++) da[i] = g[i] * scalar;
            a.AccumulateGrad(da);
        });
        return result;
    }

    public static Tensor Neg(Tensor a) => Scale(a, -1f);

    public static Tensor Sub(Tensor a, Tensor b) => Add(a, Neg(b));

    public static Tensor Sqrt(Tensor a)
    {
        var result = new Tensor(a.Shape);
        for (int i = 0; i < a.Length; i++) result.Data[i] = MathF.Sqrt(a.Data[i]);

        result.AttachGraph(new[] { a }, self =>
        {
            if (!a.RequiresGrad) return;
            var g = self.Grad!;
            var da = new float[g.Length];
            for (int i = 0; i < da.Length; i++) da[i] = g[i] * 0.5f / self.Data[i];
            a.AccumulateGrad(da);
        });
        return result;
    }

    public static Tensor Div(Tensor a, Tensor b)
    {
        var outShape = Shapes.Broadcast(a.Shape, b.Shape);
        var aExp = BroadcastSupport.ExpandToShape(a.Data, a.Shape, outShape);
        var bExp = BroadcastSupport.ExpandToShape(b.Data, b.Shape, outShape);
        var result = new Tensor(outShape);
        for (int i = 0; i < result.Length; i++) result.Data[i] = aExp[i] / bExp[i];

        result.AttachGraph(new[] { a, b }, self =>
        {
            var g = self.Grad!;
            if (a.RequiresGrad)
            {
                var da = new float[g.Length];
                for (int i = 0; i < da.Length; i++) da[i] = g[i] / bExp[i];
                a.AccumulateGrad(BroadcastSupport.ReduceGradToShape(da, outShape, a.Shape));
            }
            if (b.RequiresGrad)
            {
                var db = new float[g.Length];
                for (int i = 0; i < db.Length; i++) db[i] = -g[i] * aExp[i] / (bExp[i] * bExp[i]);
                b.AccumulateGrad(BroadcastSupport.ReduceGradToShape(db, outShape, b.Shape));
            }
        });
        return result;
    }
}
