using System;

[Serializable]
public struct MySerializableVector2
{
    public int Square;
    public int X;
    public int Y;

    public override bool Equals(object obj)
    {
        return obj is MySerializableVector2 vector &&
               Square == vector.Square &&
               X == vector.X &&
               Y == vector.Y;
    }

    public override int GetHashCode()
    {
        var hashCode = -255268455;
        hashCode = hashCode * -1521134295 + Square.GetHashCode();
        hashCode = hashCode * -1521134295 + X.GetHashCode();
        hashCode = hashCode * -1521134295 + Y.GetHashCode();
        return hashCode;
    }
}
