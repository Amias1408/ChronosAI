using Chronos.Tests;
using Chronos.Tests.EndToEnd;
using Chronos.Tests.ModelTests;
using Chronos.Tests.OpsTests;

var runner = new TestRunner();

ElementwiseTests.Register(runner);
MatMulTests.Register(runner);
ShapeOpsTests.Register(runner);
ActivationTests.Register(runner);
EmbeddingTests.Register(runner);
DropoutTests.Register(runner);

ModuleShapeTests.Register(runner);
GptModelGradCheckTest.Register(runner);

TokenizerTests.Register(runner);
CheckpointTests.Register(runner);

OverfitSanityTest.Register(runner);

return runner.RunAll();
