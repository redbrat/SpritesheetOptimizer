using System;

public sealed class Chunk
{
    public (int x, int y, int width, int height) AtlasCoordinates => _atlasCoordinatesFunc(_atlasIndex);
    public readonly int X;
    public readonly int Y;

    private readonly int _atlasIndex;
    private readonly Func<int, (int x, int y, int width, int height)> _atlasCoordinatesFunc;

    public Chunk(Func<int, (int x, int y, int width, int height)> atlasCoordinatesFunc, int atlasIndex, int x, int y)
    {
        _atlasCoordinatesFunc = atlasCoordinatesFunc;
        X = x;
        Y = y;
    }
}
