namespace Chronos.Tests;

public sealed class TestRunner
{
    private readonly List<(string Name, Action Test)> _tests = new();

    public void Add(string name, Action test) => _tests.Add((name, test));

    public int RunAll()
    {
        int passed = 0, failed = 0;
        foreach (var (name, test) in _tests)
        {
            try
            {
                test();
                Console.WriteLine($"PASS  {name}");
                passed++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FAIL  {name}: {ex.Message}");
                failed++;
            }
        }
        Console.WriteLine($"\n{passed} passed, {failed} failed, {_tests.Count} total.");
        return failed == 0 ? 0 : 1;
    }
}
