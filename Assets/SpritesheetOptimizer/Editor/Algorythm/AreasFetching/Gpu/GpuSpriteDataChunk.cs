using System;
using UnityEngine;

public class GpuSpriteDataChunk : IDisposable
{
    public readonly int SpriteWidth;
    public readonly int SpriteHeight;
    public readonly int MultipliedPart;
    public readonly int BufferSize;

    public readonly int GroupsCountX;
    public readonly int GroupsCountY;
    public readonly int GroupsCountZ;
    public readonly ComputeBuffer SpriteBuffer;
    public readonly ComputeBuffer ResultBuffer;

    public MyColor[][] OriginalSprite { get; private set; }
    public MyColor[][] Result { get; private set; }

    public GpuSpriteDataChunk(MyColor[][] sprite, uint groupSizeX, uint groupSizeY, uint groupSizeZ)
    {
        OriginalSprite = sprite;

        SpriteWidth = sprite.Length;
        SpriteHeight = sprite[0].Length;

        MultipliedPart = SpriteHeight;

        GroupsCountX = Mathf.CeilToInt((float)SpriteWidth / groupSizeX);
        GroupsCountY = Mathf.CeilToInt((float)SpriteHeight / groupSizeY);
        GroupsCountZ = 1;

        BufferSize = SpriteWidth * SpriteHeight;
        SpriteBuffer = new ComputeBuffer(BufferSize, 4);
        var spriteInts = toInts(sprite);
        SpriteBuffer.SetData(spriteInts);

        ResultBuffer = new ComputeBuffer(BufferSize, 4);
    }

    private int[] toInts(MyColor[][] sprite)
    {
        var result = new int[BufferSize];
        for (int x = 0; x < sprite.Length; x++)
            for (int y = 0; y < sprite[x].Length; y++)
                result[x * MultipliedPart + y] = sprite[x][y].Color;
        return result;
    }

    private MyColor[][] toColors(int[] ints)
    {
        var result = new MyColor[SpriteWidth][];
        for (int i = 0; i < ints.Length; i++)
        {
            var x = i / MultipliedPart;
            var y = i % MultipliedPart;
            var value = ints[i];
            var bytes = BitConverter.GetBytes(value);
            if (result[x] == null)
                result[x] = new MyColor[SpriteHeight];
            result[x][y] = new MyColor(bytes[0], bytes[1], bytes[2], bytes[3], value);
        }
        return result;
    }

    public void Dispose()
    {
        SpriteBuffer?.Dispose();
        ResultBuffer?.Dispose();
    }

    internal void FetchResults()
    {
        var ints = new int[BufferSize];
        ResultBuffer.GetData(ints);
        Result = toColors(ints);
    }
}