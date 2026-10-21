using Chronos.Core.Rng;
using Chronos.Model;
using Chronos.Tokenizer;
using Chronos.Training;

namespace Chronos.Tests.EndToEnd;

/// <summary>The single most valuable test in the whole suite: proves the full
/// pipeline (tokenizer -> model -> train -> generate -> checkpoint) is wired
/// correctly end to end, which isolated unit tests can't guarantee on their
/// own. A tiny model should be able to memorize a handful of short, distinct
/// sentences within a few hundred steps.</summary>
public static class OverfitSanityTest
{
    public static void Register(TestRunner runner)
    {
        runner.Add("End-to-end: tiny model overfits a tiny corpus", Run);
    }

    private static void Run()
    {
        RngProvider.Seed(42); // deterministic regardless of what ran earlier in the suite

        var sentences = new[]
        {
            "the cat sat on the mat",
            "the dog ran in the park",
            "a bird flew over the house",
            "the sun set behind the hill",
            "a fish swam under the bridge",
        };

        var tokenizer = BpeTokenizer.Train(sentences, new TokenizerTrainOptions(220));

        var trainingLines = new List<string>();
        for (int i = 0; i < 60; i++) trainingLines.AddRange(sentences);
        var dataset = Dataset.FromText(trainingLines, tokenizer);

        var config = new GptConfig
        {
            VocabSize = tokenizer.VocabSize,
            NLayer = 2,
            NEmbd = 32,
            NHead = 2,
            BlockSize = 32,
            DropoutP = 0f,
        };
        var model = new GptModel(config);

        var options = new TrainerOptions
        {
            MaxSteps = 1000,
            BatchSize = 8,
            WarmupSteps = 30,
            MaxLr = 2e-3f,
            MinLr = 1e-4f,
        };
        var trainer = new Trainer(model, dataset, options);

        float initialLoss = trainer.RunStep();
        float expectedInitial = MathF.Log(config.VocabSize);
        Assert.Near(expectedInitial, initialLoss, expectedInitial * 0.5f,
            "step-0 loss should be roughly ln(vocabSize) for an untrained model");

        float finalLoss = initialLoss;
        for (int i = 1; i < options.MaxSteps; i++) finalLoss = trainer.RunStep();

        Assert.True(finalLoss < initialLoss * 0.2f,
            $"expected final loss < 20% of initial ({initialLoss:F3}), got {finalLoss:F3}");
        Assert.True(finalLoss < 0.5f, $"expected final loss < 0.5, got {finalLoss:F3}");

        var promptIds = tokenizer.Encode("the cat sat on");
        var outputIds = model.Generate(promptIds, maxNewTokens: 8, temperature: 1e-6f, topK: 1, eosId: BpeTokenizer.EosId);
        var decoded = tokenizer.Decode(outputIds);
        Assert.True(decoded.Contains("the cat sat on the mat"),
            $"expected greedy generation to reproduce the memorized sentence, got '{decoded}'");

        var (logitsBefore, _) = model.Forward(promptIds, 1, promptIds.Length, targets: null, training: false);
        var before = (float[])logitsBefore.Data.Clone();

        var dir = Path.Combine(Path.GetTempPath(), $"chronos_overfit_ckpt_{Guid.NewGuid():N}");
        try
        {
            var metadata = new CheckpointMetadata { Config = config, TokenizerPath = "unused.json", Step = trainer.Step };
            Checkpoint.Save(dir, model, metadata, trainer.Optimizer);

            var (reloaded, _) = Checkpoint.Load(dir);
            var (logitsAfter, _) = reloaded.Forward(promptIds, 1, promptIds.Length, targets: null, training: false);
            Assert.ArrayNear(before, logitsAfter.Data, 1e-5f, "logits after checkpoint round-trip");
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }
}
