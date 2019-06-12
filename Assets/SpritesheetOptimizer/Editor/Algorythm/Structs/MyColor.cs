public struct MyColor
{
    private const short _byteMax = 256;
    private const int _2bytesMax = 65_536;
    private const long _4bytesMax = 4_294_967_296L;

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

        _hash = (R + G * _byteMax + B * _2bytesMax + A * _4bytesMax).GetHashCode();
    }

    public override int GetHashCode() => _hash;
}