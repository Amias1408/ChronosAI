using Chronos.Core.Rng;
using Chronos.Tokenizer;

namespace Chronos.Training;

/// <summary>A single flat token stream sampled as random contiguous windows for
/// the causal-LM objective (predict the next token at every position).</summary>
public sealed class Dataset
{
    private readonly int[] _tokens;

    public Dataset(int[] tokens)
    {
        if (tokens.Length < 2)
            throw new ArgumentException("Dataset needs at least 2 tokens.");
        _tokens = tokens;
    }

    public int Length => _tokens.Length;

    public static Dataset FromText(IEnumerable<string> lines, BpeTokenizer tokenizer)
    {
        var ids = new List<int>();
        foreach (var line in lines)
        {
            ids.AddRange(tokenizer.Encode(line));
            ids.Add(BpeTokenizer.EosId);
        }
        return new Dataset(ids.ToArray());
    }

    /// <summary>inputs/targets are flat [batchSize*blockSize]; targets[i] is the
    /// token that follows inputs[i] (a one-position shift of the same window).</summary>
    public (int[] Inputs, int[] Targets) SampleBatch(int batchSize, int blockSize)
    {
        if (_tokens.Length <= blockSize + 1)
            throw new InvalidOperationException(
                $"Dataset has {_tokens.Length} tokens, too few for blockSize {blockSize}.");

        var inputs = new int[batchSize * blockSize];
        var targets = new int[batchSize * blockSize];
        for (int b = 0; b < batchSize; b++)
        {
            int start = RngProvider.NextInt(0, _tokens.Length - blockSize - 1);
            Array.Copy(_tokens, start, inputs, b * blockSize, blockSize);
            Array.Copy(_tokens, start + 1, targets, b * blockSize, blockSize);
        }
        return (inputs, targets);
    }
}
