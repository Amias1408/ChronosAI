using System.Text.Json;
using System.Text.Json.Nodes;

namespace Chronos.Cli.Commands;

/// <summary>Points Axiom's and SCP-079-35's existing OllamaClient at Chronos
/// instead of real Ollama — a settings-file edit, not a code change, since
/// Chronos's `serve` command answers the same HTTP shape they already speak.
/// Preserves every other field already in those files (temperature, context
/// tokens, etc.); only Enabled/Host/Model are touched.</summary>
public static class ConnectCommand
{
    private static readonly (string App, string FileName)[] KnownApps =
    {
        ("Axiom", ".axiom_cs_llm.json"),
        ("SCP-079-35", ".scp079_cs_ollama.json"),
    };

    public static int Run(string[] args)
    {
        var p = new ArgParser(args);
        var port = p.GetInt("port", 11435);
        var checkpointDir = p.Get("checkpoint");
        var modelName = ModelNameFor(checkpointDir);
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

        Console.WriteLine($"Pointing local Ollama-client apps at Chronos (http://localhost:{port}, model \"{modelName}\")...");
        foreach (var (app, fileName) in KnownApps)
            ConnectOne(app, Path.Combine(home, fileName), port, modelName);

        Console.WriteLine();
        Console.WriteLine("Done. Run `chronos serve" + (checkpointDir is null ? " --checkpoint <dir>" : $" --checkpoint {checkpointDir}") +
                           $" --port {port}` and those apps will use Chronos the next time they start.");
        return 0;
    }

    public static string ModelNameFor(string? checkpointDir) =>
        checkpointDir is null ? "chronos" : $"chronos:{Path.GetFileName(Path.TrimEndingDirectorySeparator(checkpointDir))}";

    public static void ConnectOne(string appName, string path, int port, string modelName)
    {
        JsonObject obj;
        if (File.Exists(path))
        {
            obj = JsonNode.Parse(File.ReadAllText(path))?.AsObject() ?? new JsonObject();
            Console.WriteLine($"  {appName}: updating existing {path}");
        }
        else
        {
            obj = new JsonObject();
            Console.WriteLine($"  {appName}: creating {path}");
        }

        obj["Enabled"] = true;
        obj["Host"] = $"http://localhost:{port}";
        obj["Model"] = modelName;

        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, obj.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }
}
