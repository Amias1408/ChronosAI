namespace Chronos.Core;

/// <summary>Suppresses autograd graph-building for the scope's lifetime.
/// Generation would otherwise build a full unused backward graph on every forward pass.</summary>
public sealed class NoGradScope : IDisposable
{
    [ThreadStatic] private static int _depth;

    public static bool IsActive => _depth > 0;

    public NoGradScope() => _depth++;

    public void Dispose() => _depth--;
}
