using System.Text;
using Chronos.Tokenizer;
using Chronos.Training;

namespace Chronos.Cli.Commands;

/// <summary>A thin REPL wrapper over Generate — not instruction-tuned or
/// RLHF'd, said plainly so expectations stay calibrated.</summary>
public static class ChatCommand
{
    public static int Run(string[] args)
    {
        var p = new ArgParser(args);
        var checkpointDir = p.Get("checkpoint") ?? throw new ArgumentException("--checkpoint is required.");
        var temperature = p.GetFloat("temperature", 0.8f);
        var topK = p.GetInt("top-k", 40);
        var maxNew = p.GetInt("max-new-tokens", 200);

        var (model, metadata) = Checkpoint.Load(checkpointDir);
        var tokenizer = BpeTokenizer.Load(metadata.TokenizerPath);

        Console.WriteLine("Chronos chat — a thin REPL over Generate(), not instruction-tuned. Empty line to exit.");
        var history = new StringBuilder();
        while (true)
        {
            Console.Write("> ");
            var line = Console.ReadLine();
            if (string.IsNullOrEmpty(line)) break;

            history.Append("user: ").Append(line).Append('\n');
            var promptText = history + "assistant:";
            var promptIds = tokenizer.Encode(promptText);
            var outputIds = model.Generate(promptIds, maxNew, temperature, topK, eosId: BpeTokenizer.EosId);
            var reply = tokenizer.Decode(outputIds[promptIds.Length..]).Trim();

            Console.WriteLine(reply);
            history.Append("assistant: ").Append(reply).Append('\n');
        }
        return 0;
    }
}
