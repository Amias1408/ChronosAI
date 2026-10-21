namespace Chronos.Tokenizer;

/// <summary>GPT-2-style reversible byte&lt;-&gt;printable-unicode mapping. Printable
/// ASCII/Latin-1 bytes map to themselves; the remaining (mostly whitespace and
/// control) bytes map to code points starting at 256, so byte-level tokens
/// never collide with the pre-tokenizer's own whitespace handling and every
/// possible byte round-trips through a single valid Unicode char.</summary>
public static class ByteLevelEncoding
{
    private static readonly char[] ByteToChar = BuildByteToChar();
    private static readonly Dictionary<char, byte> CharToByte = BuildCharToByte();

    private static char[] BuildByteToChar()
    {
        var bytesList = new List<int>();
        for (int b = '!'; b <= '~'; b++) bytesList.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) bytesList.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) bytesList.Add(b);

        var known = new HashSet<int>(bytesList);
        var codepoints = new List<int>(bytesList);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (known.Contains(b)) continue;
            bytesList.Add(b);
            codepoints.Add(256 + n);
            n++;
        }

        var map = new char[256];
        for (int i = 0; i < bytesList.Count; i++)
            map[bytesList[i]] = (char)codepoints[i];
        return map;
    }

    private static Dictionary<char, byte> BuildCharToByte()
    {
        var d = new Dictionary<char, byte>();
        for (int b = 0; b < 256; b++) d[ByteToChar[b]] = (byte)b;
        return d;
    }

    public static string ByteToCharString(int b) => ByteToChar[b].ToString();

    public static string BytesToString(ReadOnlySpan<byte> bytes)
    {
        var chars = new char[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) chars[i] = ByteToChar[bytes[i]];
        return new string(chars);
    }

    public static byte[] StringToBytes(string s)
    {
        var bytes = new byte[s.Length];
        for (int i = 0; i < s.Length; i++) bytes[i] = CharToByte[s[i]];
        return bytes;
    }
}
