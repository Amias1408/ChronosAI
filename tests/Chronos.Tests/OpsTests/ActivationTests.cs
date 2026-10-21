using Chronos.Core.Ops;

namespace Chronos.Tests.OpsTests;

public static class ActivationTests
{
    public static void Register(TestRunner runner)
    {
        runner.Add("SoftmaxOp.Apply", () => GradCheck.CheckUnary(SoftmaxOp.Apply, new[] { 3, 5 }));
        runner.Add("SoftmaxOp.Apply 3D", () => GradCheck.CheckUnary(SoftmaxOp.Apply, new[] { 2, 3, 5 }));
        runner.Add("GeluOp.Apply", () => GradCheck.CheckUnary(GeluOp.Apply, new[] { 3, 4 }));
        runner.Add("LayerNormOp.Apply", () =>
            GradCheck.CheckTernary((x, g, b) => LayerNormOp.Apply(x, g, b), new[] { 3, 4 }, new[] { 4 }, new[] { 4 }));
    }
}
