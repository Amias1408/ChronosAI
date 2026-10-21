using Chronos.Cli;
using Chronos.Cli.Commands;

if (args.Length == 0)
    return InteractiveMenu.Run();

var command = args[0];
var rest = args[1..];
try
{
    return command switch
    {
        "tokenizer-train" => TokenizerTrainCommand.Run(rest),
        "train" => TrainCommand.Run(rest),
        "generate" => GenerateCommand.Run(rest),
        "chat" => ChatCommand.Run(rest),
        "serve" => ServeCommand.Run(rest),
        "connect" => ConnectCommand.Run(rest),
        "export-gguf" => ExportGgufCommand.Run(rest),
        "menu" => InteractiveMenu.Run(),
        "help" or "--help" or "-h" => Ok(),
        _ => Unknown(command),
    };
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Error: {ex.Message}");
    return 2;
}

int Ok()
{
    PrintUsage();
    return 0;
}

int Unknown(string cmd)
{
    Console.Error.WriteLine($"Unknown command '{cmd}'.");
    PrintUsage();
    return 1;
}

void PrintUsage()
{
    Console.WriteLine("""
        chronos                          launches the interactive menu (train, generate, chat, serve, connect)

        chronos tokenizer-train --input <dir-or-file> --vocab-size 8000 --output tokenizer.json
        chronos train --tokenizer tokenizer.json --data <corpus-dir> [--out checkpoints/run1] [--max-steps 2000] [--batch-size 16] [--resume checkpoints/run1]
          [--n-layer 4] [--n-embd 128] [--n-head 4] [--block-size 128] [--dropout 0.1] [--warmup-steps 100] [--lr 3e-4] [--checkpoint-every 500]
        chronos generate --checkpoint checkpoints/run1 --prompt "..." [--max-new-tokens 100] [--temperature 0.8] [--top-k 40]
        chronos chat --checkpoint checkpoints/run1
        chronos serve --checkpoint checkpoints/run1 [--port 11435]
        chronos connect [--checkpoint checkpoints/run1] [--port 11435]   points Axiom/SCP-079-35 at Chronos instead of Ollama
        chronos export-gguf --checkpoint checkpoints/run1 [--name chronos] [--output path.gguf] [--no-ollama-create] [--raw]
          exports to GGUF and registers it as a real Ollama model (`ollama create`), not just the mimicking HTTP server
          wraps bare input as "user: ...\nassistant:" by default so `ollama run` works for chat; --raw skips that for pure completion
        """);
}
