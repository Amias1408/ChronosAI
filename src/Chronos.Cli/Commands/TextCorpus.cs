namespace Chronos.Cli.Commands;

internal static class TextCorpus
{
    public static List<string> ReadLines(string path)
    {
        var lines = new List<string>();
        if (Directory.Exists(path))
        {
            foreach (var file in Directory.EnumerateFiles(path, "*.txt", SearchOption.AllDirectories))
                lines.AddRange(File.ReadAllLines(file));
        }
        else if (File.Exists(path))
        {
            lines.AddRange(File.ReadAllLines(path));
        }
        else
        {
            throw new FileNotFoundException($"Path not found: {path}");
        }
        return lines;
    }
}
