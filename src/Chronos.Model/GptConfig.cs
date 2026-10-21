namespace Chronos.Model;

public sealed class GptConfig
{
    public int VocabSize { get; set; }
    public int NLayer { get; set; } = 4;
    public int NEmbd { get; set; } = 128;
    public int NHead { get; set; } = 4;
    public int BlockSize { get; set; } = 128;
    public float DropoutP { get; set; } = 0.1f;

    public static GptConfig Small(int vocabSize) => new()
    {
        VocabSize = vocabSize,
        NLayer = 4,
        NEmbd = 128,
        NHead = 4,
        BlockSize = 128,
        DropoutP = 0.1f,
    };
}
