namespace Chronos.Core.Backends;

/// <summary>Ambient "current device" — the one seam ops go through, so swapping
/// or adding a backend never touches Chronos.Core.Ops.</summary>
public static class TensorBackendContext
{
    public static ITensorBackend Current { get; private set; } = new CpuTensorBackend();

    public static void Use(ITensorBackend backend) => Current = backend;
}
