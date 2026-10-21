using Chronos.Core;
using Chronos.Core.Ops;

namespace Chronos.Model;

/// <summary>Pre-LN residual block — trains more stably than post-LN at small scale.</summary>
public sealed class TransformerBlock : Module
{
    public readonly LayerNorm Ln1;
    public readonly MultiHeadSelfAttention Attn;
    public readonly LayerNorm Ln2;
    public readonly FeedForward Ffn;

    public TransformerBlock(int nEmbd, int nHead, int blockSize, float dropoutP, int nLayer)
    {
        Ln1 = new LayerNorm(nEmbd);
        Attn = new MultiHeadSelfAttention(nEmbd, nHead, blockSize, dropoutP, nLayer);
        Ln2 = new LayerNorm(nEmbd);
        Ffn = new FeedForward(nEmbd, 4, dropoutP, nLayer);
    }

    public Tensor Forward(Tensor x, bool training)
    {
        x = Elementwise.Add(x, Attn.Forward(Ln1.Forward(x), training));
        x = Elementwise.Add(x, Ffn.Forward(Ln2.Forward(x), training));
        return x;
    }

    public override IEnumerable<Tensor> Parameters()
    {
        foreach (var p in Ln1.Parameters()) yield return p;
        foreach (var p in Attn.Parameters()) yield return p;
        foreach (var p in Ln2.Parameters()) yield return p;
        foreach (var p in Ffn.Parameters()) yield return p;
    }

    public override IEnumerable<(string Name, Tensor Param)> NamedParameters(string prefix = "")
    {
        foreach (var np in Ln1.NamedParameters(prefix + "ln1.")) yield return np;
        foreach (var np in Attn.NamedParameters(prefix + "attn.")) yield return np;
        foreach (var np in Ln2.NamedParameters(prefix + "ln2.")) yield return np;
        foreach (var np in Ffn.NamedParameters(prefix + "ffn.")) yield return np;
    }
}
