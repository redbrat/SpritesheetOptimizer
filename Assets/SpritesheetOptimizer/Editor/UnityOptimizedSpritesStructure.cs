using UnityEngine;

public class UnityOptimizedSpritesStructure : ScriptableObject
{
    public Sprite[] Sprites;
    public MyVector2Float[] Pivots;
    public MyVector2Float[] Sizes;
    public SpriteChunkArrayWrapper[] Chunks;
}
