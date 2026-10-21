using Chronos.Core.Ops;

namespace Chronos.Tests.OpsTests;

public static class MatMulTests
{
    public static void Register(TestRunner runner)
    {
        runner.Add("MatMulOp.Linear 2D", () =>
            GradCheck.CheckBinary(MatMulOp.Linear, new[] { 3, 4 }, new[] { 4, 5 }));
        runner.Add("MatMulOp.Linear 3D (flattened batch)", () =>
            GradCheck.CheckBinary(MatMulOp.Linear, new[] { 2, 3, 4 }, new[] { 4, 5 }));
        runner.Add("MatMulOp.LinearTiedTransposed", () =>
            GradCheck.CheckBinary(MatMulOp.LinearTiedTransposed, new[] { 3, 4 }, new[] { 5, 4 }));
        runner.Add("MatMulOp.Batched", () =>
            GradCheck.CheckBinary(MatMulOp.Batched, new[] { 2, 3, 4 }, new[] { 2, 4, 5 }));
        runner.Add("MatMulOp.Batched 4D (multi-head shape)", () =>
            GradCheck.CheckBinary(MatMulOp.Batched, new[] { 2, 3, 4, 5 }, new[] { 2, 3, 5, 6 }));
    }
}
