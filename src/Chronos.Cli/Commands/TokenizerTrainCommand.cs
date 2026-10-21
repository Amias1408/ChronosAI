using Chronos.Tokenizer;

namespace Chronos.Cli.Commands;

public static class TokenizerTrainCommand
{
    public static int Run(string[] args)
    {
        var p = new ArgParser(args);
        var input = p.Get("input") ?? throw new ArgumentException("--input is required.");
        var output = p.Get("output", "tokenizer.json");
        var vocabSize = p.GetInt("vocab-size", 4000);

        var lines = TextCorpus.ReadLines(input);
        Console.WriteLine($"Training tokenizer on {lines.Count} lines, target vocab size {vocabSize}...");

        var tokenizer = BpeTokenizer.Train(lines, new TokenizerTrainOptions(vocabSize));
        tokenizer.Save(output);
        Console.WriteLine($"Saved tokenizer ({tokenizer.VocabSize} tokens) to {output}");
        return 0;
    }
}
