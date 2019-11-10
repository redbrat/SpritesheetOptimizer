using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class SpriteChunk
{
    public ColorsReference ColorsReference;
    public Sprite ChunkSprite;
    [NonSerialized]
    public MySerializableColor[][] Colors;
    public MySerializableAreaCoordinates Area;

    public SpriteChunk(MySerializableColor[][] colors, MyAreaCoordinates info)
    {
        Colors = colors;

        Area = new MySerializableAreaCoordinates();
        Area.X = info.X;
        Area.Y = info.Y;
        Area.Width = info.Width;
        Area.Height = info.Height;
        Area.SpriteIndex = info.SpriteIndex;
        Area.Dimensions = new MySerializableVector2();
        Area.Dimensions.X = info.Dimensions.X;
        Area.Dimensions.Y = info.Dimensions.Y;
        Area.Dimensions.Square = info.Dimensions.Square;
    }

    public override bool Equals(object obj)
    {
        return obj is SpriteChunk chunk &&
               EqualityComparer<ColorsReference>.Default.Equals(ColorsReference, chunk.ColorsReference) &&
               EqualityComparer<Sprite>.Default.Equals(ChunkSprite, chunk.ChunkSprite) &&
               EqualityComparer<MySerializableColor[][]>.Default.Equals(Colors, chunk.Colors) &&
               EqualityComparer<MySerializableAreaCoordinates>.Default.Equals(Area, chunk.Area);
    }

    public override int GetHashCode()
    {
        var hashCode = -136029698;
        hashCode = hashCode * -1521134295 + EqualityComparer<Sprite>.Default.GetHashCode(ChunkSprite);
        hashCode = hashCode * -1521134295 + EqualityComparer<MySerializableAreaCoordinates>.Default.GetHashCode(Area);
        return hashCode;
    }
}
