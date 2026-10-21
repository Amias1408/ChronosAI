using System.Text;

namespace Chronos.Tokenizer;

public static class BpeTrainer
{
    /// <summary>Learns byte-pair merges from a corpus. Recomputes pair counts from
    /// scratch every iteration — simple and correct at the small vocab/corpus sizes
    /// this project trains on; incremental pair-count updates are a future perf pass.</summary>
    public static List<(string A, string B)> Train(IEnumerable<string> corpusLines, int vocabSize)
    {
        var wordFreq = new Dictionary<string, int>();
        foreach (var line in corpusLines)
        {
            foreach (var word in PreTokenizer.Split(line))
            {
                var byteWord = ByteLevelEncoding.BytesToString(Encoding.UTF8.GetBytes(word));
                wordFreq.TryGetValue(byteWord, out var c);
                wordFreq[byteWord] = c + 1;
            }
        }

        var wordSymbols = new List<List<string>>(wordFreq.Count);
        var wordCounts = new List<int>(wordFreq.Count);
        foreach (var (byteWord, count) in wordFreq)
        {
            var symbols = new List<string>(byteWord.Length);
            foreach (var ch in byteWord) symbols.Add(ch.ToString());
            wordSymbols.Add(symbols);
            wordCounts.Add(count);
        }

        const int specialTokenCount = 3;
        const int byteTokenCount = 256;
        int targetMerges = Math.Max(0, vocabSize - specialTokenCount - byteTokenCount);
        var merges = new List<(string A, string B)>();

        while (merges.Count < targetMerges)
        {
            var pairCounts = new Dictionary<(string, string), int>();
            for (int w = 0; w < wordSymbols.Count; w++)
            {
                var symbols = wordSymbols[w];
                int freq = wordCounts[w];
                for (int i = 0; i < symbols.Count - 1; i++)
                {
                    var pair = (symbols[i], symbols[i + 1]);
                    pairCounts.TryGetValue(pair, out var c);
                    pairCounts[pair] = c + freq;
                }
            }
            if (pairCounts.Count == 0) break;

            (string A, string B) best = ("", "");
            int bestCount = -1;
            foreach (var (pair, count) in pairCounts)
            {
                if (count > bestCount ||
                    (count == bestCount && string.CompareOrdinal(pair.Item1 + pair.Item2, best.A + best.B) < 0))
                {
                    best = pair;
                    bestCount = count;
                }
            }
            merges.Add(best);

            for (int w = 0; w < wordSymbols.Count; w++)
                wordSymbols[w] = MergePair(wordSymbols[w], best.A, best.B);
        }

        return merges;
    }

    private static List<string> MergePair(List<string> symbols, string a, string b)
    {
        if (symbols.Count < 2) return symbols;
        var result = new List<string>(symbols.Count);
        int i = 0;
        while (i < symbols.Count)
        {
            if (i < symbols.Count - 1 && symbols[i] == a && symbols[i + 1] == b)
            {
                result.Add(a + b);
                i += 2;
            }
            else
            {
                result.Add(symbols[i]);
                i += 1;
            }
        }
        return result;
    }
}
