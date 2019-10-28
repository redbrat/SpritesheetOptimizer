using System;

[Serializable]
public struct MySerializableAreaCoordinates
{
    public int SpriteIndex;
    public int X;
    public int Y;
    public int Width;
    public int Height;
    public MySerializableVector2 Dimensions;
}
