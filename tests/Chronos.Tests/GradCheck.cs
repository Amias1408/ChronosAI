using Chronos.Core;
using Chronos.Core.Ops;

namespace Chronos.Tests;

/// <summary>Numerical gradient checking — the core correctness tool, since there's
/// no trusted autograd library in this codebase to compare against. Reduces any
/// op's output to a scalar via Sum(output * fixed random weights), which exercises
/// the full backward path through whatever shape the op produces.</summary>
public static class GradCheck
{
    private const float Eps = 1e-3f;
    private const float DefaultTol = 5e-2f;

    public static void CheckUnary(Func<Tensor, Tensor> f, int[] shape, float tol = DefaultTol, float offset = 0f)
    {
        var input = Tensor.Randn(shape, 1f, requiresGrad: true);
        if (offset != 0f) for (int i = 0; i < input.Length; i++) input.Data[i] += offset;
        var output = f(input);
        var weights = Tensor.Randn(output.Shape, 1f);

        var loss = Reduce.Sum(Elementwise.Mul(output, weights));
        loss.Backward();
        var analytic = (float[])input.Grad!.Clone();

        float ComputeLoss()
        {
            using var _ = new NoGradScope();
            var o = f(input);
            float sum = 0f;
            for (int i = 0; i < o.Length; i++) sum += o.Data[i] * weights.Data[i];
            return sum;
        }

        PerturbAndCheck(input, analytic, ComputeLoss, "unary", tol);
    }

    public static void CheckBinary(Func<Tensor, Tensor, Tensor> f, int[] shapeA, int[] shapeB,
        float tol = DefaultTol, float offsetA = 0f, float offsetB = 0f)
    {
        var a = Tensor.Randn(shapeA, 1f, requiresGrad: true);
        var b = Tensor.Randn(shapeB, 1f, requiresGrad: true);
        if (offsetA != 0f) for (int i = 0; i < a.Length; i++) a.Data[i] += offsetA;
        if (offsetB != 0f) for (int i = 0; i < b.Length; i++) b.Data[i] += offsetB;
        var output = f(a, b);
        var weights = Tensor.Randn(output.Shape, 1f);

        var loss = Reduce.Sum(Elementwise.Mul(output, weights));
        loss.Backward();
        var analyticA = (float[])a.Grad!.Clone();
        var analyticB = (float[])b.Grad!.Clone();

        float ComputeLoss()
        {
            using var _ = new NoGradScope();
            var o = f(a, b);
            float sum = 0f;
            for (int i = 0; i < o.Length; i++) sum += o.Data[i] * weights.Data[i];
            return sum;
        }

        PerturbAndCheck(a, analyticA, ComputeLoss, "binaryA", tol);
        PerturbAndCheck(b, analyticB, ComputeLoss, "binaryB", tol);
    }

    public static void CheckTernary(Func<Tensor, Tensor, Tensor, Tensor> f, int[] shapeA, int[] shapeB, int[] shapeC,
        float tol = DefaultTol)
    {
        var a = Tensor.Randn(shapeA, 1f, requiresGrad: true);
        var b = Tensor.Randn(shapeB, 1f, requiresGrad: true);
        var c = Tensor.Randn(shapeC, 1f, requiresGrad: true);
        var output = f(a, b, c);
        var weights = Tensor.Randn(output.Shape, 1f);

        var loss = Reduce.Sum(Elementwise.Mul(output, weights));
        loss.Backward();
        var analyticA = (float[])a.Grad!.Clone();
        var analyticB = (float[])b.Grad!.Clone();
        var analyticC = (float[])c.Grad!.Clone();

        float ComputeLoss()
        {
            using var _ = new NoGradScope();
            var o = f(a, b, c);
            float sum = 0f;
            for (int i = 0; i < o.Length; i++) sum += o.Data[i] * weights.Data[i];
            return sum;
        }

        PerturbAndCheck(a, analyticA, ComputeLoss, "ternaryA", tol);
        PerturbAndCheck(b, analyticB, ComputeLoss, "ternaryB", tol);
        PerturbAndCheck(c, analyticC, ComputeLoss, "ternaryC", tol);
    }

    private static void PerturbAndCheck(Tensor target, float[] analytic, Func<float> computeLoss, string label, float tol)
    {
        for (int i = 0; i < target.Length; i++)
        {
            float orig = target.Data[i];

            target.Data[i] = orig + Eps;
            float lossPlus = computeLoss();

            target.Data[i] = orig - Eps;
            float lossMinus = computeLoss();

            target.Data[i] = orig;

            float numeric = (lossPlus - lossMinus) / (2 * Eps);
            Assert.Near(numeric, analytic[i], tol, $"{label} grad[{i}]");
        }
    }
}
