using Chronos.Tokenizer;
using Chronos.Training;

namespace Chronos.Cli.Commands;

public static class GenerateCommand
{
    public static int Run(string[] args)
    {
        var p = new ArgParser(args);
        var checkpointDir = p.Get("checkpoint") ?? throw new ArgumentException("--checkpoint is required.");
        var prompt = p.Get("prompt", "");
        var maxNew = p.GetInt("max-new-tokens", 100);
        var temperature = p.GetFloat("temperature", 0.8f);
        var topK = p.GetInt("top-k", 40);

        var (model, metadata) = Checkpoint.Load(checkpointDir);
        var tokenizer = BpeTokenizer.Load(metadata.TokenizerPath);

        var promptIds = tokenizer.Encode(prompt);
        var outputIds = model.Generate(promptIds, maxNew, temperature, topK, eosId: BpeTokenizer.EosId);
        Console.WriteLine(tokenizer.Decode(outputIds));
        return 0;
    }
}
