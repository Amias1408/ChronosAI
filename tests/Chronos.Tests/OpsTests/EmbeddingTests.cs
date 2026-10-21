using Chronos.Core.Ops;

namespace Chronos.Tests.OpsTests;

public static class EmbeddingTests
{
    public static void Register(TestRunner runner)
    {
        runner.Add("EmbeddingOp.Lookup (with repeated index -> scatter-add)", () =>
        {
            var indices = new[] { 0, 2, 1, 2 };
            GradCheck.CheckUnary(table => EmbeddingOp.Lookup(table, indices, new[] { 4 }), new[] { 5, 3 });
        });

        runner.Add("CrossEntropyOp.Apply", () =>
        {
            var targets = new[] { 1, 0, 2 };
            GradCheck.CheckUnary(logits => CrossEntropyOp.Apply(logits, targets), new[] { 3, 4 });
        });
    }
}
