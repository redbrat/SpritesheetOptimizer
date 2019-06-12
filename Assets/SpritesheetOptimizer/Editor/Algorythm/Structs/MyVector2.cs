public struct MyVector2
{
    public static MyVector2 Zero => new MyVector2(0, 0);
    public static MyVector2 One => new MyVector2(1, 1);

    public readonly int Square;

    public readonly int X;
    public readonly int Y;

    public MyVector2(int x, int y)
    {
        X = x;
        Y = y;

        Square = x * y;
    }

    public override string ToString() => $"{X}, {Y}";
}