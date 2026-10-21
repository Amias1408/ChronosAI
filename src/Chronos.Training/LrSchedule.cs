namespace Chronos.Training;

/// <summary>Linear warmup then cosine decay (the nanoGPT recipe).</summary>
public static class LrSchedule
{
    public static float Get(int step, int warmupSteps, int maxSteps, float maxLr, float minLr)
    {
        if (warmupSteps > 0 && step < warmupSteps)
            return maxLr * (step + 1) / warmupSteps;
        if (step >= maxSteps)
            return minLr;

        float progress = (float)(step - warmupSteps) / MathF.Max(1, maxSteps - warmupSteps);
        float coeff = 0.5f * (1f + MathF.Cos(MathF.PI * progress));
        return minLr + coeff * (maxLr - minLr);
    }
}
