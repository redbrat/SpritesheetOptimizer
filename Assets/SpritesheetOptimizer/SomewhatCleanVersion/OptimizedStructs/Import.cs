using UnityEditor;
using UnityEngine;

public class Import : ScriptableObject
{
    public DefaultAsset Folder;
    public Sprite[] Chunks;
    public OptimizedSprite[] OptimizedSprites;
    public FormatMeta Meta;
}
