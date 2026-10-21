using Chronos.Core;

namespace Chronos.Training;

/// <summary>Decoupled weight decay (applied directly to Data, not folded into
/// the gradient) is what makes this AdamW rather than Adam+L2. 1-D params
/// (biases, LayerNorm gamma/beta) skip decay.</summary>
public sealed class AdamW
{
    private readonly List<Tensor> _parameters;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _eps;
    private readonly float _weightDecay;
    private readonly Dictionary<Tensor, float[]> _m = new();
    private readonly Dictionary<Tensor, float[]> _v = new();
    private int _step;

    public float LearningRate { get; set; }

    public AdamW(IEnumerable<Tensor> parameters, float lr,
        float beta1 = 0.9f, float beta2 = 0.95f, float eps = 1e-8f, float weightDecay = 0.1f)
    {
        _parameters = parameters.ToList();
        LearningRate = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = eps;
        _weightDecay = weightDecay;
    }

    public void ZeroGrad()
    {
        foreach (var p in _parameters) p.ZeroGrad();
    }

    public void Step()
    {
        _step++;
        float bc1 = 1f - MathF.Pow(_beta1, _step);
        float bc2 = 1f - MathF.Pow(_beta2, _step);

        foreach (var p in _parameters)
        {
            if (p.Grad is null) continue;
            if (!_m.TryGetValue(p, out var m)) { m = new float[p.Length]; _m[p] = m; }
            if (!_v.TryGetValue(p, out var v)) { v = new float[p.Length]; _v[p] = v; }

            bool decay = p.Rank > 1;
            for (int i = 0; i < p.Length; i++)
            {
                float g = p.Grad[i];
                m[i] = _beta1 * m[i] + (1 - _beta1) * g;
                v[i] = _beta2 * v[i] + (1 - _beta2) * g * g;
                float mHat = m[i] / bc1;
                float vHat = v[i] / bc2;
                if (decay) p.Data[i] -= LearningRate * _weightDecay * p.Data[i];
                p.Data[i] -= LearningRate * mHat / (MathF.Sqrt(vHat) + _eps);
            }
        }
    }

    public void SaveState(string path)
    {
        using var fs = File.Create(path);
        using var w = new BinaryWriter(fs);
        w.Write(_step);
        w.Write(_parameters.Count);
        foreach (var p in _parameters)
        {
            if (!_m.TryGetValue(p, out var m)) m = new float[p.Length];
            if (!_v.TryGetValue(p, out var v)) v = new float[p.Length];
            foreach (var f in m) w.Write(f);
            foreach (var f in v) w.Write(f);
        }
    }

    public void LoadState(string path)
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);
        _step = r.ReadInt32();
        int count = r.ReadInt32();
        if (count != _parameters.Count)
            throw new InvalidDataException($"Optimizer state has {count} parameters, model has {_parameters.Count}.");

        foreach (var p in _parameters)
        {
            var m = new float[p.Length];
            var v = new float[p.Length];
            for (int i = 0; i < p.Length; i++) m[i] = r.ReadSingle();
            for (int i = 0; i < p.Length; i++) v[i] = r.ReadSingle();
            _m[p] = m;
            _v[p] = v;
        }
    }
}
