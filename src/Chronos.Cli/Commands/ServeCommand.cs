using Chronos.Server;
using Chronos.Tokenizer;
using Chronos.Training;

namespace Chronos.Cli.Commands;

public static class ServeCommand
{
    public static int Run(string[] args)
    {
        var p = new ArgParser(args);
        var checkpointDir = p.Get("checkpoint") ?? throw new ArgumentException("--checkpoint is required.");
        var port = p.GetInt("port", 11435);

        var (model, metadata) = Checkpoint.Load(checkpointDir);
        var tokenizer = BpeTokenizer.Load(metadata.TokenizerPath);
        var modelName = ConnectCommand.ModelNameFor(checkpointDir);

        using var server = new OllamaCompatServer(model, tokenizer, modelName, port);

        // Ctrl+C stops the listener and returns control to the caller (the
        // interactive menu) instead of killing the whole process.
        ConsoleCancelEventHandler onCancel = (_, e) =>
        {
            e.Cancel = true;
            server.Dispose();
        };
        Console.CancelKeyPress += onCancel;
        try
        {
            server.Run();
        }
        finally
        {
            Console.CancelKeyPress -= onCancel;
        }
        return 0;
    }
}
