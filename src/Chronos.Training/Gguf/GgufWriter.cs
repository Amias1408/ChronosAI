using System.Text;

namespace Chronos.Training.Gguf;

internal enum GgufValueType : uint
{
    UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3, UInt32 = 4, Int32 = 5,
    Float32 = 6, Bool = 7, String = 8, Array = 9, UInt64 = 10, Int64 = 11, Float64 = 12,
}

/// <summary>Low-level writer for the GGUF container format (magic "GGUF",
/// version 3) — the binary layout llama.cpp/Ollama load models from. Byte
/// layout verified against llama.cpp's own gguf_writer.py: header, then
/// length-prefixed key/typed-value metadata pairs, then tensor-info records
/// (dims written in *reverse* order — ggml's ne[0] is the fastest-varying,
/// i.e. last, logical dimension), then 32-byte-aligned raw tensor data in
/// the same order the tensor-info records were declared.</summary>
internal sealed class GgufWriter
{
    private const uint Version = 3;
    private const int Alignment = 32;

    private readonly List<(string Key, byte[] Value)> _kv = new();
    private readonly List<(string Name, int[] Shape, float[] Data)> _tensors = new();

    public void AddString(string key, string value) => _kv.Add((key, Pack(GgufValueType.String, EncodeString(value))));

    public void AddUInt32(string key, uint value) => _kv.Add((key, Pack(GgufValueType.UInt32, BitConverter.GetBytes(value))));

    public void AddFloat32(string key, float value) => _kv.Add((key, Pack(GgufValueType.Float32, BitConverter.GetBytes(value))));

    public void AddStringArray(string key, IReadOnlyList<string> values) =>
        _kv.Add((key, PackArray(GgufValueType.String, values.Select(EncodeString).ToList())));

    public void AddInt32Array(string key, IReadOnlyList<int> values) =>
        _kv.Add((key, PackArray(GgufValueType.Int32, values.Select(v => BitConverter.GetBytes(v)).ToList())));

    /// <summary>shape is the logical row-major shape (e.g. [rows,cols] for a
    /// matrix already in ggml's "out,in" convention); data is F32, row-major.</summary>
    public void AddTensor(string name, int[] shape, float[] data) => _tensors.Add((name, shape, data));

    public void Save(string path)
    {
        using var fs = File.Create(path);
        using var w = new BinaryWriter(fs);

        w.Write((byte)'G'); w.Write((byte)'G'); w.Write((byte)'U'); w.Write((byte)'F');
        w.Write(Version);
        w.Write((ulong)_tensors.Count);
        w.Write((ulong)_kv.Count);

        foreach (var (key, valueBytes) in _kv)
        {
            WriteRawString(w, key);
            w.Write(valueBytes);
        }

        long offset = 0;
        foreach (var (name, shape, data) in _tensors)
        {
            WriteRawString(w, name);
            w.Write((uint)shape.Length);
            for (int i = shape.Length - 1; i >= 0; i--)
                w.Write((ulong)shape[i]);
            w.Write((uint)0); // GGML_TYPE_F32
            w.Write((ulong)offset);
            offset += PadUp((long)data.Length * sizeof(float));
        }

        WritePadding(w, fs.Position);

        foreach (var (_, _, data) in _tensors)
        {
            foreach (var f in data) w.Write(f);
            WritePadding(w, (long)data.Length * sizeof(float));
        }
    }

    private static void WriteRawString(BinaryWriter w, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static byte[] EncodeString(string s)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        WriteRawString(w, s);
        return ms.ToArray();
    }

    private static byte[] Pack(GgufValueType type, byte[] valueBytes)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write((uint)type);
        w.Write(valueBytes);
        return ms.ToArray();
    }

    private static byte[] PackArray(GgufValueType elementType, List<byte[]> elements)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write((uint)GgufValueType.Array);
        w.Write((uint)elementType);
        w.Write((ulong)elements.Count);
        foreach (var e in elements) w.Write(e);
        return ms.ToArray();
    }

    private static long PadUp(long n) => ((n + Alignment - 1) / Alignment) * Alignment;

    private static void WritePadding(BinaryWriter w, long n)
    {
        long pad = PadUp(n) - n;
        for (long i = 0; i < pad; i++) w.Write((byte)0);
    }
}
