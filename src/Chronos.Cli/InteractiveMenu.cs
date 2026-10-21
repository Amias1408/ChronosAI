using Chronos.Cli.Commands;
using Chronos.Model;
using Chronos.Tokenizer;
using Chronos.Training;

namespace Chronos.Cli;

/// <summary>The standalone terminal experience: launch `chronos` with no
/// arguments and get a menu instead of having to remember subcommand flags.
/// Every option is a thin prompt layer over the same commands the scriptable
/// CLI uses (`chronos train ...` etc.), so the two paths never drift apart.</summary>
public static class InteractiveMenu
{
    public static int Run()
    {
        PrintBanner();
        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("  1) Train a tokenizer");
            Console.WriteLine("  2) Train the model (live progress)");
            Console.WriteLine("  3) Generate text");
            Console.WriteLine("  4) Chat");
            Console.WriteLine("  5) Serve (Ollama-compatible API)");
            Console.WriteLine("  6) Connect Axiom / SCP-079-35 to Chronos");
            Console.WriteLine("  7) Export to GGUF and register as a real Ollama model");
            Console.WriteLine("  8) Exit");
            Console.Write("\n> ");
            var raw = Console.ReadLine();
            if (raw is null) return 0; // stdin closed (EOF) — exit rather than spin
            var choice = raw.Trim().ToLowerInvariant();

            try
            {
                switch (choice)
                {
                    case "1": TokenizerTrainCommand.Run(PromptTokenizerTrainArgs()); break;
                    case "2": RunTrainFlow(); break;
                    case "3": GenerateCommand.Run(PromptGenerateArgs()); break;
                    case "4": ChatCommand.Run(PromptChatArgs()); break;
                    case "5": ServeCommand.Run(PromptServeArgs()); break;
                    case "6": ConnectCommand.Run(PromptConnectArgs()); break;
                    case "7": ExportGgufCommand.Run(PromptExportGgufArgs()); break;
                    case "8" or "exit" or "quit" or "q": return 0;
                    case "": break;
                    default: Console.WriteLine("Not a valid option."); break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }
    }

    private static void PrintBanner()
    {
        Console.WriteLine("=== Chronos — a self-hosted language model, trained entirely on this machine ===");
        Console.WriteLine("No Ollama, no pretrained weights, no external packages — everything it knows");
        Console.WriteLine("comes from data you give it and training that happens right here.");
    }

    // --- Train ---------------------------------------------------------

    private static void RunTrainFlow()
    {
        var tokenizerPath = Ask("Tokenizer path", FindDefaultTokenizer());
        if (!File.Exists(tokenizerPath))
        {
            Console.WriteLine($"No tokenizer found at '{tokenizerPath}'.");
            if (AskYesNo("Train one now from a data folder?", true))
            {
                var input = Ask("Training text folder or file", "data/seed");
                var vocab = Ask("Target vocab size", "4000");
                TokenizerTrainCommand.Run(new[] { "--input", input, "--output", tokenizerPath, "--vocab-size", vocab });
            }
            else
            {
                return;
            }
        }

        var dataPath = Ask("Training data path", "data/seed");
        var outDir = AskCheckpoint("Checkpoint directory", "checkpoints/chronos");

        var args = new List<string> { "--tokenizer", tokenizerPath, "--data", dataPath, "--out", outDir };
        bool isExisting = File.Exists(Path.Combine(outDir, "checkpoint.json"));

        if (isExisting && AskYesNo($"Found an existing checkpoint at '{outDir}' — resume it?", true))
        {
            args.AddRange(new[] { "--resume", outDir });
        }
        else if (!isExisting)
        {
            var fast = AskYesNo("Use the fast CPU-friendly preset (2 layers)? No = the larger default preset (4 layers, slower)", true);
            if (fast)
                args.AddRange(new[] { "--n-layer", "2", "--n-embd", "64", "--n-head", "2", "--block-size", "64" });
        }

        args.AddRange(new[] { "--max-steps", Ask("Max steps", "2000"), "--batch-size", Ask("Batch size", "8") });
        Console.WriteLine();
        Console.WriteLine("Training — Ctrl+C stops early and saves a checkpoint.");
        TrainCommand.Run(args.ToArray());
    }

    // --- Other commands --------------------------------------------------

    private static string[] PromptTokenizerTrainArgs()
    {
        var input = Ask("Training text folder or file", "data/seed");
        var output = Ask("Tokenizer output path", "data/seed/tokenizer.json");
        var vocab = Ask("Target vocab size", "4000");
        return new[] { "--input", input, "--output", output, "--vocab-size", vocab };
    }

    private static string[] PromptGenerateArgs()
    {
        var checkpoint = AskCheckpoint("Checkpoint to generate from", "checkpoints/chronos");
        var prompt = Ask("Prompt", "");
        var maxNew = Ask("Max new tokens", "100");
        var temperature = Ask("Temperature", "0.8");
        return new[] { "--checkpoint", checkpoint, "--prompt", prompt, "--max-new-tokens", maxNew, "--temperature", temperature };
    }

    private static string[] PromptChatArgs()
    {
        var checkpoint = AskCheckpoint("Checkpoint to chat with", "checkpoints/chronos");
        return new[] { "--checkpoint", checkpoint };
    }

    private static string[] PromptServeArgs()
    {
        var checkpoint = AskCheckpoint("Checkpoint to serve", "checkpoints/chronos");
        var port = Ask("Port", "11435");
        Console.WriteLine("Serving — Ctrl+C stops and returns to this menu.");
        return new[] { "--checkpoint", checkpoint, "--port", port };
    }

    private static string[] PromptConnectArgs()
    {
        var checkpoint = AskCheckpoint("Checkpoint Axiom/SCP-079-35 should use", "checkpoints/chronos");
        var port = Ask("Port Chronos will serve on", "11435");
        return new[] { "--checkpoint", checkpoint, "--port", port };
    }

    private static string[] PromptExportGgufArgs()
    {
        var checkpoint = AskCheckpoint("Checkpoint to export", "checkpoints/chronos");
        var name = Ask("Name to register in Ollama", Path.GetFileName(Path.TrimEndingDirectorySeparator(checkpoint)));
        return new[] { "--checkpoint", checkpoint, "--name", name };
    }

    // --- Prompt helpers --------------------------------------------------

    private static string Ask(string prompt, string fallback)
    {
        Console.Write(string.IsNullOrEmpty(fallback) ? $"{prompt}: " : $"{prompt} [{fallback}]: ");
        var line = Console.ReadLine();
        return string.IsNullOrWhiteSpace(line) ? fallback : line.Trim();
    }

    private static bool AskYesNo(string prompt, bool defaultYes)
    {
        var answer = Ask($"{prompt} (y/n)", defaultYes ? "y" : "n");
        return answer.StartsWith("y", StringComparison.OrdinalIgnoreCase);
    }

    private static string AskCheckpoint(string prompt, string fallback)
    {
        var checkpoints = ListCheckpoints();
        if (checkpoints.Length == 0) return Ask(prompt, fallback);

        Console.WriteLine($"{prompt}:");
        for (int i = 0; i < checkpoints.Length; i++)
            Console.WriteLine($"    {i + 1}) {checkpoints[i]}");
        Console.Write($"  Pick a number, or type a path [{fallback}]: ");
        var line = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(line)) return fallback;
        return int.TryParse(line, out var idx) && idx >= 1 && idx <= checkpoints.Length
            ? checkpoints[idx - 1]
            : line;
    }

    private static string[] ListCheckpoints()
    {
        if (!Directory.Exists("checkpoints")) return Array.Empty<string>();
        return Directory.EnumerateDirectories("checkpoints")
            .Where(d => File.Exists(Path.Combine(d, "checkpoint.json")))
            .OrderByDescending(Directory.GetLastWriteTimeUtc)
            .ToArray();
    }

    private static string FindDefaultTokenizer()
    {
        const string preferred = "data/seed/tokenizer.json";
        if (File.Exists(preferred)) return preferred;
        if (Directory.Exists("data"))
        {
            var found = Directory.EnumerateFiles("data", "tokenizer.json", SearchOption.AllDirectories).FirstOrDefault();
            if (found is not null) return found;
        }
        return preferred;
    }
}
