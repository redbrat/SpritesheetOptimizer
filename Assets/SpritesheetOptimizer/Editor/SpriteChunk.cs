using System;
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
}
