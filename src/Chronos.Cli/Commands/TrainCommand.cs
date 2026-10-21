using Chronos.Model;
using Chronos.Tokenizer;
using Chronos.Training;

namespace Chronos.Cli.Commands;

public static class TrainCommand
{
    public static int Run(string[] args)
    {
        var p = new ArgParser(args);
        var tokenizerPath = p.Get("tokenizer") ?? throw new ArgumentException("--tokenizer is required.");
        var dataPath = p.Get("data") ?? throw new ArgumentException("--data is required.");
        var outDir = p.Get("out", "checkpoints/run1");
        var resume = p.Get("resume");

        var tokenizer = BpeTokenizer.Load(tokenizerPath);
        var lines = TextCorpus.ReadLines(dataPath);
        var dataset = Dataset.FromText(lines, tokenizer);
        Console.WriteLine($"Loaded {dataset.Length} tokens from {dataPath}.");

        GptModel model;
        CheckpointMetadata metadata;
        int startStep = 0;

        if (resume is not null)
        {
            (model, metadata) = Checkpoint.Load(resume);
            startStep = metadata.Step;
            Console.WriteLine($"Resumed from {resume} at step {startStep}.");
        }
        else
        {
            var defaults = GptConfig.Small(tokenizer.VocabSize);
            var config = new GptConfig
            {
                VocabSize = tokenizer.VocabSize,
                NLayer = p.GetInt("n-layer", defaults.NLayer),
                NEmbd = p.GetInt("n-embd", defaults.NEmbd),
                NHead = p.GetInt("n-head", defaults.NHead),
                BlockSize = p.GetInt("block-size", defaults.BlockSize),
                DropoutP = p.GetFloat("dropout", defaults.DropoutP),
            };
            model = new GptModel(config);
            metadata = new CheckpointMetadata { Config = config, TokenizerPath = Path.GetFullPath(tokenizerPath) };
        }

        var options = new TrainerOptions
        {
            MaxSteps = p.GetInt("max-steps", 2000),
            BatchSize = p.GetInt("batch-size", 16),
            WarmupSteps = p.GetInt("warmup-steps", 100),
            MaxLr = p.GetFloat("lr", 3e-4f),
            CheckpointEvery = p.GetInt("checkpoint-every", 500),
        };

        return RunTraining(model, dataset, options, metadata, outDir, resume);
    }

    /// <summary>Shared by the CLI subcommand and the interactive menu so both
    /// paths exercise (and benefit from) the same live progress view and the
    /// same graceful-stop-on-Ctrl+C behavior.</summary>
    public static int RunTraining(GptModel model, Dataset dataset, TrainerOptions options,
        CheckpointMetadata metadata, string outDir, string? resume)
    {
        var trainer = new Trainer(model, dataset, options, metadata.Step);
        if (resume is not null)
        {
            var optStatePath = Path.Combine(resume, "checkpoint.optstate.bin");
            if (File.Exists(optStatePath)) trainer.Optimizer.LoadState(optStatePath);
        }

        var stopRequested = false;
        ConsoleCancelEventHandler onCancel = (_, e) =>
        {
            e.Cancel = true; // don't let the process die mid-step; finish this step and save
            stopRequested = true;
        };

        var view = new LiveTrainingView(options.MaxSteps, options.BatchSize * model.Config.BlockSize);
        Console.CancelKeyPress += onCancel;
        try
        {
            while (trainer.Step < options.MaxSteps && !stopRequested)
            {
                float loss = trainer.RunStep();
                view.Update(trainer.Step, loss, trainer.Optimizer.LearningRate);

                if (trainer.Step % options.CheckpointEvery == 0)
                {
                    metadata.Step = trainer.Step;
                    Checkpoint.Save(outDir, model, metadata, trainer.Optimizer);
                    view.Note($"checkpoint saved to {outDir} at step {trainer.Step}");
                }
            }
        }
        finally
        {
            Console.CancelKeyPress -= onCancel;
        }

        metadata.Step = trainer.Step;
        Checkpoint.Save(outDir, model, metadata, trainer.Optimizer);
        view.Finish(stopRequested);
        return 0;
    }
}
