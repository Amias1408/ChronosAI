namespace Chronos.Tokenizer;

/// <summary>On-disk tokenizer file. The 256 byte tokens and the 3 special tokens
/// are always the same deterministic set (see <see cref="ByteLevelEncoding"/> and
/// <see cref="BpeTokenizer.SpecialTokens"/>) so only the learned merges need saving.</summary>
public sealed class TokenizerFile
{
    public int VocabSize { get; set; }
    public List<string[]> Merges { get; set; } = new();
}

public sealed record TokenizerTrainOptions(int VocabSize);
