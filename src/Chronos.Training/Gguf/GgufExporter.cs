using Chronos.Model;
using Chronos.Tokenizer;

namespace Chronos.Training.Gguf;

/// <summary>Exports a Chronos checkpoint as a GGUF file llama.cpp/Ollama can
/// load natively under the "gpt2" architecture. This works at all because
/// Chronos's transformer was deliberately built to GPT-2's exact shape (fused
/// QKV projection, pre-LN blocks, tied embeddings, GELU MLP) — tensor names,
/// metadata keys and the weight-transpose convention below were checked
/// directly against llama.cpp's own conversion/gpt2.py and gguf-py/constants.py,
/// not guessed.</summary>
public static class GgufExporter
{
    public static void Export(GptModel model, BpeTokenizer tokenizer, string outputPath, string modelName)
    {
        var cfg = model.Config;
        var w = new GgufWriter();

        w.AddString("general.architecture", "gpt2");
        w.AddString("general.name", modelName);
        w.AddUInt32("general.file_type", 0); // ALL_F32

        w.AddUInt32("gpt2.context_length", (uint)cfg.BlockSize);
        w.AddUInt32("gpt2.embedding_length", (uint)cfg.NEmbd);
        w.AddUInt32("gpt2.block_count", (uint)cfg.NLayer);
        w.AddUInt32("gpt2.attention.head_count", (uint)cfg.NHead);
        w.AddFloat32("gpt2.attention.layer_norm_epsilon", 1e-5f);
        w.AddUInt32("gpt2.feed_forward_length", (uint)(4 * cfg.NEmbd));

        var (tokens, tokenTypes, merges) = tokenizer.ExportVocabForGguf();
        w.AddString("tokenizer.ggml.model", "gpt2");
        w.AddString("tokenizer.ggml.pre", "gpt-2");
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddInt32Array("tokenizer.ggml.token_type", tokenTypes);
        w.AddStringArray("tokenizer.ggml.merges", merges);
        w.AddUInt32("tokenizer.ggml.bos_token_id", BpeTokenizer.BosId);
        w.AddUInt32("tokenizer.ggml.eos_token_id", BpeTokenizer.EosId);
        w.AddUInt32("tokenizer.ggml.padding_token_id", BpeTokenizer.PadId);

        // token_embd / output are weight-tied in Chronos — write the same data twice,
        // exactly like HF's GPT2LMHeadModel state_dict does for a tied lm_head.
        w.AddTensor("token_embd.weight", model.TokenEmbedding.Weight.Shape, model.TokenEmbedding.Weight.Data);
        w.AddTensor("output.weight", model.TokenEmbedding.Weight.Shape, model.TokenEmbedding.Weight.Data);
        w.AddTensor("position_embd.weight", model.PositionEmbedding.Weight.Shape, model.PositionEmbedding.Weight.Data);

        w.AddTensor("output_norm.weight", model.LnFinal.Gamma.Shape, model.LnFinal.Gamma.Data);
        w.AddTensor("output_norm.bias", model.LnFinal.Beta.Shape, model.LnFinal.Beta.Data);

        for (int i = 0; i < model.Blocks.Count; i++)
        {
            var block = model.Blocks[i];
            var p = $"blk.{i}.";

            w.AddTensor(p + "attn_norm.weight", block.Ln1.Gamma.Shape, block.Ln1.Gamma.Data);
            w.AddTensor(p + "attn_norm.bias", block.Ln1.Beta.Shape, block.Ln1.Beta.Data);

            AddLinearTransposed(w, p + "attn_qkv", block.Attn.QkvProj);
            AddLinearTransposed(w, p + "attn_output", block.Attn.OutProj);

            w.AddTensor(p + "ffn_norm.weight", block.Ln2.Gamma.Shape, block.Ln2.Gamma.Data);
            w.AddTensor(p + "ffn_norm.bias", block.Ln2.Beta.Shape, block.Ln2.Beta.Data);

            AddLinearTransposed(w, p + "ffn_up", block.Ffn.Fc1);
            AddLinearTransposed(w, p + "ffn_down", block.Ffn.Fc2);
        }

        w.Save(outputPath);
    }

    /// <summary>Chronos stores Linear weights as [inDim,outDim] (so forward is a
    /// plain x@W); ggml/GGUF wants the standard nn.Linear convention,
    /// [outDim,inDim]. Biases need no transpose (1-D).</summary>
    private static void AddLinearTransposed(GgufWriter w, string baseName, Linear linear)
    {
        int inDim = linear.Weight.Shape[0], outDim = linear.Weight.Shape[1];
        var transposed = Transpose2D(linear.Weight.Data, inDim, outDim);
        w.AddTensor(baseName + ".weight", new[] { outDim, inDim }, transposed);
        if (linear.Bias is not null)
            w.AddTensor(baseName + ".bias", linear.Bias.Shape, linear.Bias.Data);
    }

    private static float[] Transpose2D(float[] data, int rows, int cols)
    {
        var result = new float[data.Length];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j * rows + i] = data[i * cols + j];
        return result;
    }
}
