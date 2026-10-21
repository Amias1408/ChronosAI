namespace Chronos.Tests;

public sealed class TestFailedException(string message) : Exception(message);

public static class Assert
{
    public static void True(bool cond, string msg)
    {
        if (!cond) throw new TestFailedException(msg);
    }

    public static void Near(float expected, float actual, float tol, string msg)
    {
        if (MathF.Abs(expected - actual) > tol)
            throw new TestFailedException($"{msg} (expected {expected}, got {actual}, tol {tol})");
    }

    public static void ArrayNear(float[] expected, float[] actual, float tol, string msg)
    {
        True(expected.Length == actual.Length, $"{msg}: length mismatch ({expected.Length} vs {actual.Length})");
        for (int i = 0; i < expected.Length; i++)
            Near(expected[i], actual[i], tol, $"{msg}[{i}]");
    }
}
