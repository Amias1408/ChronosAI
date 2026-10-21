using Chronos.Core;
using Chronos.Core.Ops;

namespace Chronos.Tests.OpsTests;

/// <summary>Dropout is stochastic, so naive finite-differencing is meaningless —
/// these check the exact forward/backward mask math instead.</summary>
public static class DropoutTests
{
    public static void Register(TestRunner runner)
    {
        runner.Add("DropoutOp eval-mode is identity", () =>
        {
            var x = Tensor.Randn(new[] { 3, 4 }, 1f, requiresGrad: true);
            var y = DropoutOp.Apply(x, 0.5f, training: false);
            Assert.True(ReferenceEquals(x, y), "eval-mode dropout should return the same tensor instance");
        });

        runner.Add("DropoutOp train-mode mask reused in backward", () =>
        {
            var x = Tensor.Randn(new[] { 200 }, 1f, requiresGrad: true);
            var y = DropoutOp.Apply(x, 0.5f, training: true);

            int zeros = 0, kept = 0;
            for (int i = 0; i < x.Length; i++)
            {
                if (y.Data[i] == 0f) { zeros++; }
                else { Assert.Near(x.Data[i] * 2f, y.Data[i], 1e-5f, $"kept[{i}]"); kept++; }
            }
            Assert.True(zeros > 0 && kept > 0, "expected a mix of dropped and kept elements over 200 draws");

            var loss = Reduce.Sum(y);
            loss.Backward();
            for (int i = 0; i < x.Length; i++)
            {
                float expected = y.Data[i] == 0f ? 0f : 2f;
                Assert.Near(expected, x.Grad![i], 1e-5f, $"grad[{i}]");
            }
        });
    }
}
