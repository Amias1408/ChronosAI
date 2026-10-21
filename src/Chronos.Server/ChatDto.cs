using System.Text.Json.Serialization;

namespace Chronos.Server;

public sealed class ChatMessageDto
{
    public string Role { get; set; } = "";
    public string Content { get; set; } = "";
}

public sealed class ChatOptionsDto
{
    [JsonPropertyName("num_ctx")]
    public int? NumCtx { get; set; }

    [JsonPropertyName("num_predict")]
    public int? NumPredict { get; set; }

    public double? Temperature { get; set; }
}

public sealed class ChatRequestDto
{
    public string? Model { get; set; }
    public List<ChatMessageDto> Messages { get; set; } = new();
    public bool Stream { get; set; } = true;
    public ChatOptionsDto? Options { get; set; }
}
