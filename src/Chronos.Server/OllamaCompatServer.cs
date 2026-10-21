using System.Net;
using System.Text;
using System.Text.Json;
using Chronos.Model;
using Chronos.Tokenizer;

namespace Chronos.Server;

/// <summary>Loopback-only HTTP server answering just enough of Ollama's API
/// (GET /api/tags, POST /api/chat) for the existing OllamaClient.cs in Axiom
/// and SCP-079-35 to work against it unmodified — a config edit, not a code
/// change, once there's a checkpoint worth pointing at.</summary>
public sealed class OllamaCompatServer : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

    private readonly HttpListener _listener;
    private readonly GptModel _model;
    private readonly BpeTokenizer _tokenizer;
    private readonly string _modelName;

    public OllamaCompatServer(GptModel model, BpeTokenizer tokenizer, string modelName, int port)
    {
        _model = model;
        _tokenizer = tokenizer;
        _modelName = modelName;
        _listener = new HttpListener();
        _listener.Prefixes.Add($"http://localhost:{port}/");
    }

    public void Run(CancellationToken ct = default)
    {
        _listener.Start();
        Console.WriteLine($"Chronos serving on {_listener.Prefixes.First()} (model \"{_modelName}\")");

        while (!ct.IsCancellationRequested)
        {
            HttpListenerContext ctx;
            try { ctx = _listener.GetContext(); }
            catch (HttpListenerException) { break; }
            catch (ObjectDisposedException) { break; }
            _ = Task.Run(() => Handle(ctx));
        }
    }

    private void Handle(HttpListenerContext ctx)
    {
        try
        {
            var path = ctx.Request.Url?.AbsolutePath ?? "";
            if (ctx.Request.HttpMethod == "GET" && path == "/api/tags")
                HandleTags(ctx);
            else if (ctx.Request.HttpMethod == "POST" && path == "/api/chat")
                HandleChat(ctx);
            else
                ctx.Response.StatusCode = 404;
        }
        catch (Exception ex)
        {
            try
            {
                ctx.Response.StatusCode = 500;
                WriteLine(ctx.Response, JsonSerializer.Serialize(new { error = ex.Message }));
            }
            catch { /* client likely disconnected */ }
        }
        finally
        {
            ctx.Response.Close();
        }
    }

    private void HandleTags(HttpListenerContext ctx)
    {
        ctx.Response.ContentType = "application/json";
        var payload = new { models = new[] { new { name = _modelName } } };
        WriteLine(ctx.Response, JsonSerializer.Serialize(payload));
    }

    private void HandleChat(HttpListenerContext ctx)
    {
        using var reader = new StreamReader(ctx.Request.InputStream, Encoding.UTF8);
        var body = reader.ReadToEnd();
        var req = JsonSerializer.Deserialize<ChatRequestDto>(body, JsonOptions) ?? new ChatRequestDto();

        var promptIds = _tokenizer.Encode(BuildPrompt(req.Messages));
        int maxNew = req.Options?.NumPredict ?? 200;
        float temperature = (float)(req.Options?.Temperature ?? 0.8);

        ctx.Response.ContentType = "application/x-ndjson";
        ctx.Response.SendChunked = true;

        _model.Generate(promptIds, maxNew, temperature, topK: 40, eosId: BpeTokenizer.EosId, onToken: id =>
        {
            var piece = _tokenizer.Decode(new[] { id });
            var chunk = new { message = new { role = "assistant", content = piece }, done = false };
            WriteLine(ctx.Response, JsonSerializer.Serialize(chunk));
        });

        WriteLine(ctx.Response, JsonSerializer.Serialize(new { message = new { role = "assistant", content = "" }, done = true }));
    }

    private static void WriteLine(HttpListenerResponse response, string json)
    {
        var bytes = Encoding.UTF8.GetBytes(json + "\n");
        response.OutputStream.Write(bytes, 0, bytes.Length);
        response.OutputStream.Flush();
    }

    private static string BuildPrompt(List<ChatMessageDto> messages)
    {
        var sb = new StringBuilder();
        foreach (var m in messages)
            sb.Append(m.Role).Append(": ").Append(m.Content).Append('\n');
        sb.Append("assistant:");
        return sb.ToString();
    }

    public void Dispose() => _listener.Close();
}
