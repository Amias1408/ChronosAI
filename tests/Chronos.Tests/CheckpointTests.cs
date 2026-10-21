using Chronos.Model;
using Chronos.Training;

namespace Chronos.Tests;

public static class CheckpointTests
{
    public static void Register(TestRunner runner)
    {
        runner.Add("Checkpoint save/load round-trip preserves logits", () =>
        {
            var config = new GptConfig { VocabSize = 12, NLayer = 1, NEmbd = 8, NHead = 2, BlockSize = 6, DropoutP = 0f };
            var model = new GptModel(config);
            var tokens = new[] { 1, 2, 3, 4 };
            var (logitsBefore, _) = model.Forward(tokens, 1, 4, targets: null, training: false);
            var before = (float[])logitsBefore.Data.Clone();

            var dir = Path.Combine(Path.GetTempPath(), $"chronos_ckpt_test_{Guid.NewGuid():N}");
            try
            {
                var metadata = new CheckpointMetadata { Config = config, TokenizerPath = "unused.json" };
                Checkpoint.Save(dir, model, metadata);

                var (loaded, loadedMetadata) = Checkpoint.Load(dir);
                var (logitsAfter, _) = loaded.Forward(tokens, 1, 4, targets: null, training: false);

                Assert.ArrayNear(before, logitsAfter.Data, 1e-6f, "logits after checkpoint round-trip");
                Assert.True(loadedMetadata.Config.VocabSize == config.VocabSize, "config mismatch after reload");
            }
            finally
            {
                if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
            }
        });
    }
}
