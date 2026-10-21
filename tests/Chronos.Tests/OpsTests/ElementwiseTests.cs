using Chronos.Core.Ops;

namespace Chronos.Tests.OpsTests;

public static class ElementwiseTests
{
    public static void Register(TestRunner runner)
    {
        runner.Add("Elementwise.Add same-shape", () =>
            GradCheck.CheckBinary(Elementwise.Add, new[] { 3, 4 }, new[] { 3, 4 }));
        runner.Add("Elementwise.Add broadcast bias", () =>
            GradCheck.CheckBinary(Elementwise.Add, new[] { 3, 4 }, new[] { 4 }));
        runner.Add("Elementwise.Add broadcast leading dims", () =>
            GradCheck.CheckBinary(Elementwise.Add, new[] { 2, 3, 4 }, new[] { 3, 4 }));
        runner.Add("Elementwise.Mul same-shape", () =>
            GradCheck.CheckBinary(Elementwise.Mul, new[] { 3, 4 }, new[] { 3, 4 }));
        runner.Add("Elementwise.Mul broadcast", () =>
            GradCheck.CheckBinary(Elementwise.Mul, new[] { 3, 4 }, new[] { 4 }));
        runner.Add("Elementwise.Scale", () =>
            GradCheck.CheckUnary(x => Elementwise.Scale(x, 2.5f), new[] { 3, 4 }));
        runner.Add("Elementwise.Neg", () =>
            GradCheck.CheckUnary(Elementwise.Neg, new[] { 3, 4 }));
        runner.Add("Elementwise.Sub", () =>
            GradCheck.CheckBinary(Elementwise.Sub, new[] { 3, 4 }, new[] { 3, 4 }));
        runner.Add("Elementwise.Sqrt", () =>
            GradCheck.CheckUnary(Elementwise.Sqrt, new[] { 3, 4 }, offset: 4f));
        runner.Add("Elementwise.Div", () =>
            GradCheck.CheckBinary(Elementwise.Div, new[] { 3, 4 }, new[] { 3, 4 }, offsetA: 0f, offsetB: 4f));
    }
}
