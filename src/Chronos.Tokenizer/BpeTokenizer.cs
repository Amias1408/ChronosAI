using System.Text;
using System.Text.Json;

namespace Chronos.Tokenizer;

/// <summary>Byte-level BPE, GPT-2 style. Every possible input round-trips (no
/// &lt;unk&gt;) because the base vocabulary is the 256 raw bytes; merges only ever
/// combine tokens that are already in the vocabulary.</summary>
public sealed class BpeTokenizer
{
    public const int BosId = 0;
    public const int EosId = 1;
    public const int PadId = 2;
    private const int SpecialTokenCount = 3;
    private const int ByteTokenCount = 256;

    public static readonly IReadOnlyDictionary<string, int> SpecialTokens = new Dictionary<string, int>
    {
        ["<|bos|>"] = BosId,
        ["<|eos|>"] = EosId,
        ["<|pad|>"] = PadId,
    };

    private readonly List<string> _idToToken;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<(string, string), int> _mergeRank;

    public int VocabSize => _idToToken.Count;

    private BpeTokenizer(List<(string A, string B)> merges)
    {
        _idToToken = new List<string>(SpecialTokenCount + ByteTokenCount + merges.Count);
        _idToToken.Add("<|bos|>");
        _idToToken.Add("<|eos|>");
        _idToToken.Add("<|pad|>");
        for (int b = 0; b < ByteTokenCount; b++)
            _idToToken.Add(ByteLevelEncoding.ByteToCharString(b));
        foreach (var (a, b) in merges)
            _idToToken.Add(a + b);

        _tokenToId = new Dictionary<string, int>(_idToToken.Count);
        for (int i = 0; i < _idToToken.Count; i++) _tokenToId[_idToToken[i]] = i;

        _mergeRank = new Dictionary<(string, string), int>(merges.Count);
        for (int i = 0; i < merges.Count; i++) _mergeRank[merges[i]] = i;
    }

    public static BpeTokenizer Train(IEnumerable<string> corpusLines, TokenizerTrainOptions options)
    {
        var lines = corpusLines as IReadOnlyCollection<string> ?? corpusLines.ToList();
        var merges = BpeTrainer.Train(lines, options.VocabSize);
        return new BpeTokenizer(merges);
    }

    public int[] Encode(string text)
    {
        var ids = new List<int>();
        foreach (var word in PreTokenizer.Split(text))
        {
            var byteWord = ByteLevelEncoding.BytesToString(Encoding.UTF8.GetBytes(word));
            foreach (var symbol in BpeEncodeWord(byteWord))
                ids.Add(_tokenToId[symbol]);
        }
        return ids.ToArray();
    }

    public string Decode(ReadOnlySpan<int> ids)
    {
        var sb = new StringBuilder();
        foreach (var id in ids)
        {
            if (id < SpecialTokenCount) continue;
            sb.Append(_idToToken[id]);
        }
        return Encoding.UTF8.GetString(ByteLevelEncoding.StringToBytes(sb.ToString()));
    }

    private List<string> BpeEncodeWord(string byteWord)
    {
        var symbols = new List<string>(byteWord.Length);
        foreach (var ch in byteWord) symbols.Add(ch.ToString());
        if (symbols.Count <= 1) return symbols;

        while (true)
        {
            int bestRank = int.MaxValue, bestIndex = -1;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_mergeRank.TryGetValue((symbols[i], symbols[i + 1]), out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) break;

            string a = symbols[bestIndex], b = symbols[bestIndex + 1];
            var merged = new List<string>(symbols.Count);
            int idx = 0;
            while (idx < symbols.Count)
            {
                if (idx < symbols.Count - 1 && symbols[idx] == a && symbols[idx + 1] == b)
                {
                    merged.Add(a + b);
                    idx += 2;
                }
                else
                {
                    merged.Add(symbols[idx]);
                    idx += 1;
                }
            }
            symbols = merged;
            if (symbols.Count == 1) break;
        }
        return symbols;
    }

    /// <summary>Vocab/merges in the shape a GGUF exporter needs: token strings
    /// in id order, a llama.cpp TokenType per id (CONTROL=3 for the three
    /// special tokens, NORMAL=1 otherwise), and "a b"-joined merges in rank
    /// order.</summary>
    public (List<string> Tokens, List<int> TokenTypes, List<string> Merges) ExportVocabForGguf()
    {
        var tokenTypes = new List<int>(_idToToken.Count);
        for (int i = 0; i < _idToToken.Count; i++)
            tokenTypes.Add(i < SpecialTokenCount ? 3 : 1); // CONTROL : NORMAL

        var merges = new List<string>(_mergeRank.Count);
        foreach (var pair in _mergeRank.OrderBy(kv => kv.Value).Select(kv => kv.Key))
            merges.Add($"{pair.Item1} {pair.Item2}");

        return (new List<string>(_idToToken), tokenTypes, merges);
    }

    public void Save(string path)
    {
        var merges = new List<string[]>(_mergeRank.Count);
        foreach (var pair in _mergeRank.OrderBy(kv => kv.Value).Select(kv => kv.Key))
            merges.Add(new[] { pair.Item1, pair.Item2 });

        var file = new TokenizerFile { VocabSize = VocabSize, Merges = merges };
        File.WriteAllText(path, JsonSerializer.Serialize(file, new JsonSerializerOptions { WriteIndented = true }));
    }

    public static BpeTokenizer Load(string path)
    {
        var file = JsonSerializer.Deserialize<TokenizerFile>(File.ReadAllText(path))
                   ?? throw new InvalidDataException($"Could not parse tokenizer file '{path}'.");
        var merges = file.Merges.Select(m => (m[0], m[1])).ToList();
        return new BpeTokenizer(merges);
    }
}
