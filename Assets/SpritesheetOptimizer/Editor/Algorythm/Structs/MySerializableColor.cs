using System;

[Serializable]
public struct MySerializableColor
{
    public int R;
    public int G;
    public int B;
    public int A;

    public int Color;

    public MySerializableColor(int color)
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
    }
}
