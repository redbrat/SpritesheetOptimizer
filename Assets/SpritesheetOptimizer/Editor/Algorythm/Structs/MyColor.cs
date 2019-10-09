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

    public MyColor(byte r, byte g, byte b, byte a)
    {
        R = r;
        G = g;
        B = b;
        A = a;

        //var ri = (int)r;
        //var gi = g << 8;
        //var bi = b << 16;
        //var ai = a << 24;
        var ri = r << 24;
        var gi = g << 16;
        var bi = b << 8;
        var ai = (int)a;
        Color = ri | gi | bi | ai;


        int ir = (Color >> 24) & 255;
        int ig = (Color >> 16) & 255;
        int ib = (Color >> 8) & 255;
        int ia = Color & 255;

        var reconstructed = (ir << 24) | (ig << 16) | (ib << 8) | ia;


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