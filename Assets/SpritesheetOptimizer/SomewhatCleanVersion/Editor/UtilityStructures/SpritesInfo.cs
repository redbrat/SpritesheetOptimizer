using UnityEngine;

public class SpritesInfo
{
    public MyColor[][][] Colors;
    public byte[][][][] Bytes;
    public Sprite[] Sprites;

    public SpritesInfo(MyColor[][][] colors, byte[][][][] bytes, Sprite[] sprites)
    {
        Colors = colors;
        Bytes = bytes;
        Sprites = sprites;
    }
}
