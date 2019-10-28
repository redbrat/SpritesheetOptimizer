using System;

[Serializable]
public class SpriteChunkArrayWrapper
{
    public SpriteChunk[] Array;

    public SpriteChunkArrayWrapper(SpriteChunk[] array)
    {
        Array = array;
    }
}
