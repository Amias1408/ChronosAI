using System.Diagnostics;
using Chronos.Tokenizer;
using Chronos.Training;
using Chronos.Training.Gguf;

namespace Chronos.Cli.Commands;

/// <summary>Exports a checkpoint to GGUF and, by default, registers it with
/// the real Ollama app (`ollama create`) so it shows up in `ollama list` /
/// `ollama run` — not just reachable through the mimicking `chronos serve`
/// HTTP server, but an actual local Ollama model.</summary>
public static class ExportGgufCommand
{
    public static int Run(string[] args)
    {
        var p = new ArgParser(args);
        var checkpointDir = p.Get("checkpoint") ?? throw new ArgumentException("--checkpoint is required.");
        var name = p.Get("name") ?? Path.GetFileName(Path.TrimEndingDirectorySeparator(checkpointDir));
        var outputPath = Path.GetFullPath(p.Get("output") ?? Path.Combine(checkpointDir, $"{name}.gguf"));
        var registerWithOllama = !p.Has("no-ollama-create");
        var rawTemplate = p.Has("raw");

        var (model, metadata) = Checkpoint.Load(checkpointDir);
        var tokenizer = BpeTokenizer.Load(metadata.TokenizerPath);

        Console.WriteLine($"Exporting '{checkpointDir}' to GGUF ({model.Config.NLayer} layers, " +
                           $"{model.Config.NEmbd} dim, vocab {tokenizer.VocabSize})...");
        GgufExporter.Export(model, tokenizer, outputPath, name);
        Console.WriteLine($"Wrote {outputPath} ({new FileInfo(outputPath).Length / 1024.0:F0} KB)");

        if (!registerWithOllama) return 0;

        // Wraps bare input the way ChatCommand and the dialogue training data both
        // do ("user: {prompt}\nassistant:"), so `ollama run` works for chat-style
        // use out of the box instead of needing "user: ...\nassistant:" typed by
        // hand every time. --raw skips this for pure literary-completion use.
        var modelfileContent = rawTemplate
            ? $"FROM {outputPath}\n"
            : $"FROM {outputPath}\n" + "TEMPLATE \"\"\"user: {{ .Prompt }}\nassistant:\"\"\"\n";

        var modelfilePath = Path.Combine(Path.GetDirectoryName(outputPath)!, "Modelfile");
        File.WriteAllText(modelfilePath, modelfileContent);
        Console.WriteLine($"Wrote {modelfilePath}");
        Console.WriteLine($"Running: ollama create {name} -f \"{modelfilePath}\"");

        try
        {
            using var proc = Process.Start(new ProcessStartInfo("ollama", $"create {name} -f \"{modelfilePath}\"")
            {
                UseShellExecute = false,
            });
            proc?.WaitForExit();
            Console.WriteLine(proc?.ExitCode == 0
                ? $"Done — try: ollama run {name}"
                : $"`ollama create` exited with code {proc?.ExitCode}. You can retry manually: ollama create {name} -f \"{modelfilePath}\"");
        }
        catch (System.ComponentModel.Win32Exception)
        {
            Console.WriteLine("Could not find `ollama` on PATH — install it, then run manually: " +
                               $"ollama create {name} -f \"{modelfilePath}\"");
        }
        return 0;
    }
}
