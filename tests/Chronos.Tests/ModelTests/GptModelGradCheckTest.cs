using Chronos.Core;
using Chronos.Model;

namespace Chronos.Tests.ModelTests;

/// <summary>The single most valuable correctness test at the model level: even
/// when every isolated op passes its own gradient check, wiring them together
/// wrong (a transposed dimension, a missed accumulation) would only show up
/// here. Dropout is off (training:false) so the two finite-difference forward
/// passes stay deterministic.</summary>
public static class GptModelGradCheckTest
{
    private const float Eps = 1e-3f;
    private const float Tol = 5e-2f;

    public static void Register(TestRunner runner)
    {
        runner.Add("GptModel full backward matches numerical gradient (tiny model)", Run);
    }

    private static void Run()
    {
        var config = new GptConfig { VocabSize = 8, NLayer = 1, NEmbd = 4, NHead = 2, BlockSize = 4, DropoutP = 0f };
        var model = new GptModel(config);
        var tokens = new[] { 1, 3, 5 };
        var targets = new[] { 3, 5, 2 };

        model.ZeroGrad();
        var (_, loss) = model.Forward(tokens, 1, 3, targets, training: false);
        loss!.Backward();

        foreach (var (name, param) in model.NamedParameters())
        {
            var analytic = (float[])param.Grad!.Clone();
            for (int i = 0; i < param.Length; i++)
            {
                float orig = param.Data[i];

                param.Data[i] = orig + Eps;
                float lossPlus = ForwardLoss(model, tokens, targets);

                param.Data[i] = orig - Eps;
                float lossMinus = ForwardLoss(model, tokens, targets);

                param.Data[i] = orig;

                float numeric = (lossPlus - lossMinus) / (2 * Eps);
                Assert.Near(numeric, analytic[i], Tol, $"{name}[{i}]");
            }
        }
    }

    private static float ForwardLoss(GptModel model, int[] tokens, int[] targets)
    {
        using var _ = new NoGradScope();
        var (_, loss) = model.Forward(tokens, 1, tokens.Length, targets, training: false);
        return loss!.Data[0];
    }
}
