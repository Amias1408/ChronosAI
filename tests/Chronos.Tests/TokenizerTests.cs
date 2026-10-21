using Chronos.Tokenizer;

namespace Chronos.Tests;

public static class TokenizerTests
{
    public static void Register(TestRunner runner)
    {
        runner.Add("BpeTokenizer encode/decode round-trip", () =>
        {
            var corpus = new[]
            {
                "the quick brown fox jumps over the lazy dog",
                "the dog barked at the fox",
                "a quick fox is quick",
            };
            var tokenizer = BpeTokenizer.Train(corpus, new TokenizerTrainOptions(300));

            foreach (var text in corpus.Append("an unseen sentence with new words: firetruck!"))
            {
                var decoded = tokenizer.Decode(tokenizer.Encode(text));
                Assert.True(decoded == text, $"round-trip mismatch: '{text}' -> '{decoded}'");
            }
        });

        runner.Add("BpeTokenizer save/load round-trip", () =>
        {
            var corpus = new[] { "hello world", "hello there", "world of tokens" };
            var tokenizer = BpeTokenizer.Train(corpus, new TokenizerTrainOptions(280));
            var path = Path.Combine(Path.GetTempPath(), $"chronos_tok_test_{Guid.NewGuid():N}.json");
            try
            {
                tokenizer.Save(path);
                var loaded = BpeTokenizer.Load(path);
                Assert.True(loaded.VocabSize == tokenizer.VocabSize, "vocab size mismatch after reload");

                foreach (var text in corpus)
                {
                    var a = tokenizer.Encode(text);
                    var b = loaded.Encode(text);
                    Assert.True(a.SequenceEqual(b), $"encode mismatch after reload for '{text}'");
                }
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        });
    }
}
