public struct MyColor
{
    private const short _byteMax = 256;
    private const int _2bytesMax = 65_536;
    private const long _4bytesMax = 4_294_967_296L;

    public int Color { get; }

    public readonly byte R;
    public readonly byte G;
    public readonly byte B;
    public readonly byte A;

    private readonly int _hash;

    public MyColor(int color)
    {
        var r = (byte)(color << 24 & 255);
        var g = (byte)(color << 16 & 255);
        var b = (byte)(color << 8 & 255);
        var a = (byte)(color & 255);

        R = r;
        G = g;
        B = b;
        A = a;

        Color = color;

        _hash = (R + G * _byteMax + B * _2bytesMax + A * _4bytesMax).GetHashCode();
    }

    public MyColor(byte r, byte g, byte b, byte a)
    {
        R = r;
        G = g;
        B = b;
        A = a;

        var ri = r << 24;
        var gi = g << 16;
        var bi = b << 8;
        var ai = (int)a;

        Color = ri | gi | bi | ai;

        _hash = (R + G * _byteMax + B * _2bytesMax + A * _4bytesMax).GetHashCode();
    }

    public MyColor(byte r, byte g, byte b, byte a, int intColor)
    {
        R = r;
        G = g;
        B = b;
        A = a;

        Color = intColor;

        _hash = (R + G * _byteMax + B * _2bytesMax + A * _4bytesMax).GetHashCode();
    }

    public override int GetHashCode() => _hash;
}