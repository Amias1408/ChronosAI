using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Chronos.Core;
using Chronos.Model;

namespace Chronos.Training;

public sealed class CheckpointMetadata
{
    public GptConfig Config { get; set; } = new();
    public string TokenizerPath { get; set; } = "";
    public int Step { get; set; }
    public float BestLoss { get; set; } = float.PositiveInfinity;
}

/// <summary>Two files per checkpoint: checkpoint.json (config + metadata, human
/// readable) and checkpoint.weights.bin (the actual floats). Weights are matched
/// back in by parameter name on load, not by position — robust to future
/// module-order changes.</summary>
public static class Checkpoint
{
    private const string Magic = "CHRONOS1";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public static void Save(string dir, GptModel model, CheckpointMetadata metadata, AdamW? optimizer = null)
    {
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "checkpoint.json"), JsonSerializer.Serialize(metadata, JsonOptions));
        SaveWeights(Path.Combine(dir, "checkpoint.weights.bin"), model);
        optimizer?.SaveState(Path.Combine(dir, "checkpoint.optstate.bin"));
    }

    public static (GptModel Model, CheckpointMetadata Metadata) Load(string dir)
    {
        var json = File.ReadAllText(Path.Combine(dir, "checkpoint.json"));
        var metadata = JsonSerializer.Deserialize<CheckpointMetadata>(json, JsonOptions)
                       ?? throw new InvalidDataException("Could not parse checkpoint.json.");
        var model = new GptModel(metadata.Config);
        LoadWeights(Path.Combine(dir, "checkpoint.weights.bin"), model);
        return (model, metadata);
    }

    private static void SaveWeights(string path, GptModel model)
    {
        using var fs = File.Create(path);
        using var w = new BinaryWriter(fs);
        w.Write(Encoding.ASCII.GetBytes(Magic));
        w.Write(1);

        var named = model.NamedParameters().ToList();
        w.Write(named.Count);
        foreach (var (name, param) in named)
        {
            var nameBytes = Encoding.UTF8.GetBytes(name);
            w.Write(nameBytes.Length);
            w.Write(nameBytes);
            w.Write(param.Shape.Length);
            foreach (var d in param.Shape) w.Write(d);
            foreach (var f in param.Data) w.Write(f);
        }
    }

    private static void LoadWeights(string path, GptModel model)
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);

        var magic = Encoding.ASCII.GetString(r.ReadBytes(Magic.Length));
        if (magic != Magic) throw new InvalidDataException($"Bad checkpoint magic '{magic}'.");
        _ = r.ReadInt32(); // version
        int count = r.ReadInt32();

        var byName = model.NamedParameters().ToDictionary(np => np.Name, np => np.Param);
        for (int i = 0; i < count; i++)
        {
            string name = Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
            int ndims = r.ReadInt32();
            var shape = new int[ndims];
            for (int d = 0; d < ndims; d++) shape[d] = r.ReadInt32();
            int n = Shapes.Numel(shape);
            var data = new float[n];
            for (int j = 0; j < n; j++) data[j] = r.ReadSingle();

            if (!byName.TryGetValue(name, out var param))
                throw new InvalidDataException($"Checkpoint has unknown parameter '{name}'.");
            if (!Shapes.ShapeEquals(param.Shape, shape))
                throw new InvalidDataException(
                    $"Checkpoint parameter '{name}' shape [{string.Join(',', shape)}] " +
                    $"does not match model shape [{string.Join(',', param.Shape)}].");
            Array.Copy(data, param.Data, n);
        }
    }
}
