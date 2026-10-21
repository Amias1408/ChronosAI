using System.Diagnostics;

namespace Chronos.Cli;

/// <summary>Renders training progress so it's easy to watch live in a terminal:
/// a single self-overwriting line when attached to a real console, or periodic
/// full lines when output is redirected (piped to a file/log, where \r is
/// meaningless).</summary>
public sealed class LiveTrainingView
{
    private readonly int _maxSteps;
    private readonly int _tokensPerStep;
    private readonly bool _interactive;
    private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
    private readonly int _logEveryFallback;

    public LiveTrainingView(int maxSteps, int tokensPerStep)
    {
        _maxSteps = Math.Max(1, maxSteps);
        _tokensPerStep = tokensPerStep;
        _interactive = !Console.IsOutputRedirected;
        _logEveryFallback = Math.Max(1, _maxSteps / 20);
    }

    public void Update(int step, float loss, float lr)
    {
        double elapsed = _stopwatch.Elapsed.TotalSeconds;
        double stepsPerSec = step / Math.Max(elapsed, 1e-6);
        double toksPerSec = stepsPerSec * _tokensPerStep;
        double etaSec = stepsPerSec > 0 ? (_maxSteps - step) / stepsPerSec : 0;

        const int barWidth = 24;
        int filled = Math.Clamp((int)(barWidth * ((double)step / _maxSteps)), 0, barWidth);
        var bar = new string('#', filled) + new string('-', barWidth - filled);

        var line = $"[{bar}] {step,6}/{_maxSteps}  loss {loss,7:F4}  lr {lr:E1}  " +
                   $"{toksPerSec,7:F0} tok/s  elapsed {FormatTime(elapsed)}  eta {FormatTime(etaSec)}";

        if (_interactive)
            Console.Write("\r" + line.PadRight(110));
        else if (step % _logEveryFallback == 0 || step == _maxSteps)
            Console.WriteLine(line);
    }

    public void Note(string message)
    {
        if (_interactive) Console.WriteLine();
        Console.WriteLine(message);
    }

    public void Finish(bool stoppedEarly)
    {
        if (_interactive) Console.WriteLine();
        Console.WriteLine(stoppedEarly ? "Training stopped early — checkpoint saved." : "Training complete.");
    }

    private static string FormatTime(double seconds)
    {
        var ts = TimeSpan.FromSeconds(Math.Max(0, seconds));
        return ts.TotalHours >= 1 ? $"{(int)ts.TotalHours}h{ts.Minutes:D2}m" : $"{ts.Minutes}m{ts.Seconds:D2}s";
    }
}
