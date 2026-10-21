namespace Chronos.Core.Rng;

/// <summary>Single seeded RNG shared by weight init and dropout so a run is
/// reproducible end to end. Not thread-safe by design — all draws happen on
/// the thread building the graph, never inside a parallel kernel.</summary>
public static class RngProvider
{
    private static System.Random _rng = new(1337);

    public static void Seed(int seed) => _rng = new System.Random(seed);

    public static float NextFloat() => (float)_rng.NextDouble();

    public static bool NextBernoulli(double p) => _rng.NextDouble() < p;

    public static int NextInt(int minInclusive, int maxExclusive) => _rng.Next(minInclusive, maxExclusive);

    /// <summary>Standard normal via Box-Muller.</summary>
    public static float NextGaussian()
    {
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = _rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }

    public static void FillGaussian(Span<float> dest, float std)
    {
        for (int i = 0; i < dest.Length; i++) dest[i] = NextGaussian() * std;
    }
}
