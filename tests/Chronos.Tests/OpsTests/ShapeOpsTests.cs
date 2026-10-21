using Chronos.Core.Ops;

namespace Chronos.Tests.OpsTests;

public static class ShapeOpsTests
{
    public static void Register(TestRunner runner)
    {
        runner.Add("ShapeOps.Reshape", () =>
            GradCheck.CheckUnary(x => ShapeOps.Reshape(x, new[] { 2, 6 }), new[] { 3, 4 }));
        runner.Add("ShapeOps.Transpose 2D", () =>
            GradCheck.CheckUnary(x => ShapeOps.Transpose(x, 0, 1), new[] { 3, 4 }));
        runner.Add("ShapeOps.Transpose 4D middle axes", () =>
            GradCheck.CheckUnary(x => ShapeOps.Transpose(x, 1, 2), new[] { 2, 3, 4, 5 }));
        runner.Add("ShapeOps.Split output 0", () =>
            GradCheck.CheckUnary(x => ShapeOps.Split(x, new[] { 2, 2 })[0], new[] { 3, 4 }));
        runner.Add("ShapeOps.Split output 1", () =>
            GradCheck.CheckUnary(x => ShapeOps.Split(x, new[] { 2, 2 })[1], new[] { 3, 4 }));
        runner.Add("ShapeOps.Concat", () =>
            GradCheck.CheckBinary((a, b) => ShapeOps.Concat(new[] { a, b }, -1), new[] { 3, 2 }, new[] { 3, 3 }));
    }
}
