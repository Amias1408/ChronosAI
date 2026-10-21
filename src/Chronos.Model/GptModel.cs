using Chronos.Core;
using Chronos.Core.Ops;
using Chronos.Core.Rng;

namespace Chronos.Model;

public sealed class GptModel : Module
{
    public readonly GptConfig Config;
    public readonly Embedding TokenEmbedding;
    public readonly PositionalEmbedding PositionEmbedding;
    public readonly List<TransformerBlock> Blocks;
    public readonly LayerNorm LnFinal;

    public GptModel(GptConfig config)
    {
        Config = config;
        TokenEmbedding = new Embedding(config.VocabSize, config.NEmbd);
        PositionEmbedding = new PositionalEmbedding(config.BlockSize, config.NEmbd);
        Blocks = new List<TransformerBlock>(config.NLayer);
        for (int i = 0; i < config.NLayer; i++)
            Blocks.Add(new TransformerBlock(config.NEmbd, config.NHead, config.BlockSize, config.DropoutP, config.NLayer));
        LnFinal = new LayerNorm(config.NEmbd);
    }

    /// <summary>tokenIds is a flat row-major [batchSize,seqLen] buffer. targets, if
    /// given, is the same shape flattened — the token that should follow each
    /// position (fused softmax+NLL is computed via <see cref="CrossEntropyOp"/>).</summary>
    public (Tensor Logits, Tensor? Loss) Forward(
        int[] tokenIds, int batchSize, int seqLen, int[]? targets = null, bool training = true)
    {
        if (seqLen > Config.BlockSize)
            throw new InvalidOperationException($"Sequence length {seqLen} exceeds BlockSize {Config.BlockSize}.");

        var tokEmb = TokenEmbedding.Forward(tokenIds, new[] { batchSize, seqLen }); // [B,T,D]
        var posEmb = PositionEmbedding.Forward(seqLen); // [T,D], broadcasts over batch
        var x = Elementwise.Add(tokEmb, posEmb);
        x = DropoutOp.Apply(x, Config.DropoutP, training);

        foreach (var block in Blocks)
            x = block.Forward(x, training);

        x = LnFinal.Forward(x);
        var logits = MatMulOp.LinearTiedTransposed(x, TokenEmbedding.Weight); // [B,T,V], weight-tied

        Tensor? loss = targets is null
            ? null
            : CrossEntropyOp.Apply(ShapeOps.Reshape(logits, new[] { batchSize * seqLen, Config.VocabSize }), targets);

        return (logits, loss);
    }

    /// <summary>Autoregressive sampling inside a NoGradScope. Prompts/continuations
    /// longer than BlockSize are truncated to the most recent BlockSize tokens
    /// rather than erroring — relevant once served over HTTP with an arbitrary
    /// chat history. eosId, if given, stops generation early.</summary>
    public int[] Generate(int[] promptIds, int maxNewTokens, float temperature, int topK,
        int? eosId = null, Action<int>? onToken = null)
    {
        using var _ = new NoGradScope();
        var ids = new List<int>(promptIds);

        for (int step = 0; step < maxNewTokens; step++)
        {
            int[] window = ids.Count > Config.BlockSize
                ? ids.GetRange(ids.Count - Config.BlockSize, Config.BlockSize).ToArray()
                : ids.ToArray();
            int t = window.Length;

            var (logits, _) = Forward(window, 1, t, targets: null, training: false);

            int v = Config.VocabSize;
            var lastLogits = new float[v];
            Array.Copy(logits.Data, (t - 1) * v, lastLogits, 0, v);

            int next = SampleNextToken(lastLogits, temperature, topK);
            ids.Add(next);
            onToken?.Invoke(next);
            if (eosId is not null && next == eosId) break;
        }
        return ids.ToArray();
    }

    private static int SampleNextToken(float[] logits, float temperature, int topK)
    {
        int v = logits.Length;
        float t = MathF.Max(temperature, 1e-6f);
        var scaled = new float[v];
        for (int i = 0; i < v; i++) scaled[i] = logits[i] / t;

        if (topK > 0 && topK < v)
        {
            var keep = Enumerable.Range(0, v).OrderByDescending(i => scaled[i]).Take(topK).ToHashSet();
            for (int i = 0; i < v; i++)
                if (!keep.Contains(i)) scaled[i] = float.NegativeInfinity;
        }

        float max = scaled.Max();
        var probs = new float[v];
        float sum = 0f;
        for (int i = 0; i < v; i++)
        {
            probs[i] = MathF.Exp(scaled[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < v; i++) probs[i] /= sum;

        float r = RngProvider.NextFloat();
        float acc = 0f;
        for (int i = 0; i < v; i++)
        {
            acc += probs[i];
            if (r <= acc) return i;
        }
        return v - 1;
    }

    public override IEnumerable<Tensor> Parameters()
    {
        foreach (var p in TokenEmbedding.Parameters()) yield return p;
        foreach (var p in PositionEmbedding.Parameters()) yield return p;
        foreach (var block in Blocks)
            foreach (var p in block.Parameters()) yield return p;
        foreach (var p in LnFinal.Parameters()) yield return p;
    }

    public override IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "")
    {
        foreach (var np in TokenEmbedding.NamedParameters(prefix + "token_embedding.")) yield return np;
        foreach (var np in PositionEmbedding.NamedParameters(prefix + "position_embedding.")) yield return np;
        for (int i = 0; i < Blocks.Count; i++)
            foreach (var np in Blocks[i].NamedParameters($"{prefix}blocks.{i}.")) yield return np;
        foreach (var np in LnFinal.NamedParameters(prefix + "ln_final.")) yield return np;
    }
}
