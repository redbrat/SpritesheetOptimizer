using System;

public sealed class Chunk
{
    public (int x, int y, int width, int height) AtlasCoordinates => _atlasCoordinatesFunc(AtlasIndex);

    public readonly int X;
    public readonly int Y;
    public readonly int AtlasIndex;

    private readonly Func<int, (int x, int y, int width, int height)> _atlasCoordinatesFunc;

    public Chunk(Func<int, (int x, int y, int width, int height)> atlasCoordinatesFunc, int atlasIndex, int x, int y)
    {
        AtlasIndex = atlasIndex;
        _atlasCoordinatesFunc = atlasCoordinatesFunc;
        X = x;
        Y = y;
    }
}
