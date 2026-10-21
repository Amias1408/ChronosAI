using Chronos.Core;
using Chronos.Model;

namespace Chronos.Tests.ModelTests;

public static class ModuleShapeTests
{
    public static void Register(TestRunner runner)
    {
        runner.Add("Linear output shape", () =>
        {
            var linear = new Linear(4, 6);
            var x = Tensor.Randn(new[] { 2, 3, 4 }, 1f);
            var y = linear.Forward(x);
            Assert.True(y.Shape.SequenceEqual(new[] { 2, 3, 6 }), $"expected [2,3,6], got [{string.Join(',', y.Shape)}]");
        });

        runner.Add("MultiHeadSelfAttention output shape", () =>
        {
            var attn = new MultiHeadSelfAttention(nEmbd: 8, nHead: 2, blockSize: 5, dropoutP: 0f, nLayer: 1);
            var x = Tensor.Randn(new[] { 2, 5, 8 }, 1f);
            var y = attn.Forward(x, training: false);
            Assert.True(y.Shape.SequenceEqual(new[] { 2, 5, 8 }), $"expected [2,5,8], got [{string.Join(',', y.Shape)}]");
        });

        runner.Add("GptModel forward logits shape", () =>
        {
            var config = new GptConfig { VocabSize = 10, NLayer = 2, NEmbd = 8, NHead = 2, BlockSize = 6, DropoutP = 0f };
            var model = new GptModel(config);
            var tokens = new[] { 1, 2, 3, 4, 5, 6, 7, 8 }; // batch=2, seq=4
            var targets = new[] { 2, 3, 4, 5, 6, 7, 8, 9 };
            var (logits, loss) = model.Forward(tokens, 2, 4, targets, training: false);
            Assert.True(logits.Shape.SequenceEqual(new[] { 2, 4, 10 }), $"expected [2,4,10], got [{string.Join(',', logits.Shape)}]");
            Assert.True(loss is not null, "expected a loss tensor when targets are supplied");
        });

        runner.Add("GptModel.Generate produces the requested token budget (or stops early)", () =>
        {
            var config = new GptConfig { VocabSize = 10, NLayer = 1, NEmbd = 4, NHead = 2, BlockSize = 8, DropoutP = 0f };
            var model = new GptModel(config);
            var output = model.Generate(new[] { 1, 2 }, maxNewTokens: 5, temperature: 1f, topK: 0);
            Assert.True(output.Length >= 2 && output.Length <= 7, $"expected between 2 and 7 tokens, got {output.Length}");
        });
    }
}
