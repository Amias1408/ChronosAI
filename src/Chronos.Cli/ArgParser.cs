namespace Chronos.Cli;

/// <summary>Hand-rolled `--flag value` / bare `--flag` parsing — no CLI package,
/// keeping the zero-dependency rule literal even in plumbing.</summary>
public sealed class ArgParser
{
    private readonly Dictionary<string, string> _values = new();

    public ArgParser(string[] args)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal)) continue;
            var key = args[i][2..];
            if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                _values[key] = args[i + 1];
                i++;
            }
            else
            {
                _values[key] = "true";
            }
        }
    }

    public string? Get(string key) => _values.TryGetValue(key, out var v) ? v : null;

    public string Get(string key, string fallback) => _values.TryGetValue(key, out var v) ? v : fallback;

    public int GetInt(string key, int fallback) =>
        _values.TryGetValue(key, out var v) && int.TryParse(v, out var i) ? i : fallback;

    public float GetFloat(string key, float fallback) =>
        _values.TryGetValue(key, out var v) && float.TryParse(v, out var f) ? f : fallback;

    public bool Has(string key) => _values.ContainsKey(key);
}
