using System.Text.RegularExpressions;

namespace Chronos.Tokenizer;

/// <summary>Splits raw text into pre-tokens — the exact GPT-2 pattern
/// (contractions, then runs of letters/digits/punctuation each with an
/// optional leading space, then trailing whitespace), byte-for-byte the same
/// regex llama.cpp hardcodes for LLAMA_VOCAB_PRE_TYPE_GPT2. Used identically
/// at BPE train time and encode time — any drift between the two would
/// silently break encoding, and any drift from llama.cpp's copy would silently
/// mistokenize once a checkpoint is exported to GGUF and run under Ollama,
/// since Ollama re-tokenizes with its own hardcoded copy of this pattern
/// rather than anything shipped in the GGUF file.</summary>
public static partial class PreTokenizer
{
    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)")]
    private static partial Regex Pattern();

    public static IEnumerable<string> Split(string text)
    {
        foreach (Match m in Pattern().Matches(text))
            yield return m.Value;
    }
}
