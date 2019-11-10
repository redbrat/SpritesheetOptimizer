using System;
using System.Collections.Generic;

[Serializable]
public struct MySerializableAreaCoordinates
{
    public int SpriteIndex;
    public int X;
    public int Y;
    public int Width;
    public int Height;
    public MySerializableVector2 Dimensions;

    public override bool Equals(object obj)
    {
        return obj is MySerializableAreaCoordinates coordinates &&
               SpriteIndex == coordinates.SpriteIndex &&
               X == coordinates.X &&
               Y == coordinates.Y &&
               Width == coordinates.Width &&
               Height == coordinates.Height &&
               EqualityComparer<MySerializableVector2>.Default.Equals(Dimensions, coordinates.Dimensions);
    }

    public override int GetHashCode()
    {
        var hashCode = -748962844;
        hashCode = hashCode * -1521134295 + SpriteIndex.GetHashCode();
        hashCode = hashCode * -1521134295 + X.GetHashCode();
        hashCode = hashCode * -1521134295 + Y.GetHashCode();
        hashCode = hashCode * -1521134295 + Width.GetHashCode();
        hashCode = hashCode * -1521134295 + Height.GetHashCode();
        hashCode = hashCode * -1521134295 + EqualityComparer<MySerializableVector2>.Default.GetHashCode(Dimensions);
        return hashCode;
    }
}
