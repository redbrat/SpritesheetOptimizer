public struct MyAreaCoordinates
{
    public readonly int SpriteIndex;
    public readonly int X;
    public readonly int Y;
    public readonly int Width;
    public readonly int Height;
    public readonly MyVector2 Dimensions;

    private readonly int _hash;

    public MyAreaCoordinates(int spriteIndex, int x, int y, int width, int height)
    {
        SpriteIndex = spriteIndex;
        X = x;
        Y = y;
        Width = width;
        Height = height;
        Dimensions = new MyVector2(width, height);

        _hash = 1 + spriteIndex;
        _hash *= Width * 100;
        _hash *= Height * 10_000;
        _hash *= (x + 1) * 1_000_000;
        _hash *= (y + 1) * 100_000_000;
    }

    public override int GetHashCode() => _hash;
}
