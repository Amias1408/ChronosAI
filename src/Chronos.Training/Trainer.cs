using Chronos.Model;

namespace Chronos.Training;

public sealed class TrainerOptions
{
    public int MaxSteps { get; set; } = 2000;
    public int BatchSize { get; set; } = 16;
    public int WarmupSteps { get; set; } = 100;
    public float MaxLr { get; set; } = 3e-4f;
    public float MinLr { get; set; } = 3e-5f;
    public float WeightDecay { get; set; } = 0.1f;
    public float GradClipNorm { get; set; } = 1.0f;
    public int LogEvery { get; set; } = 50;
    public int CheckpointEvery { get; set; } = 500;
}

public sealed class Trainer
{
    private readonly GptModel _model;
    private readonly Dataset _dataset;
    private readonly TrainerOptions _options;

    public AdamW Optimizer { get; }
    public int Step { get; private set; }
    public float LastLoss { get; private set; } = float.NaN;

    public Trainer(GptModel model, Dataset dataset, TrainerOptions options, int startStep = 0)
    {
        _model = model;
        _dataset = dataset;
        _options = options;
        Optimizer = new AdamW(model.Parameters(), options.MaxLr, weightDecay: options.WeightDecay);
        Step = startStep;
    }

    public float RunStep()
    {
        Optimizer.LearningRate = LrSchedule.Get(Step, _options.WarmupSteps, _options.MaxSteps, _options.MaxLr, _options.MinLr);

        var (inputs, targets) = _dataset.SampleBatch(_options.BatchSize, _model.Config.BlockSize);
        _model.ZeroGrad();
        var (_, loss) = _model.Forward(inputs, _options.BatchSize, _model.Config.BlockSize, targets, training: true);
        loss!.Backward();
        GradClip.ClipGlobalNorm(_model.Parameters(), _options.GradClipNorm);
        Optimizer.Step();

        LastLoss = loss.Data[0];
        Step++;
        return LastLoss;
    }

    public void Run(Action<int, float>? onLog = null, Action<int>? onCheckpoint = null)
    {
        while (Step < _options.MaxSteps)
        {
            float loss = RunStep();
            if (onLog is not null && Step % _options.LogEvery == 0) onLog(Step, loss);
            if (onCheckpoint is not null && Step % _options.CheckpointEvery == 0) onCheckpoint(Step);
        }
    }
}
