using System.Collections.Generic;
using UnityEngine;

public class OptimizedSprite : ScriptableObject
{
    private static List<Color> _colors = new List<Color>();

    public OptimizedSpriteChunk[] Chunks;
    public Vector2 Pivot;
    public int Width;
    public int Height;

    public Sprite OriginalSprite;

    public void DebugSprite()
    {
        var newTex = new Texture2D(Width, Height, TextureFormat.ARGB32, false, false);
        newTex.filterMode = FilterMode.Point;
        for (int x = 0; x < Width; x++)
        {
            for (int y = 0; y < Height; y++)
            {
                newTex.SetPixel(x, y, GetPixel(x, y));
            }
        }

        newTex.Apply();
        var newSprite = Sprite.Create(newTex, new Rect(Vector2.zero, new Vector2(Width, Height)), Vector2.one * 0.5f, 100, 0, SpriteMeshType.FullRect);
        OriginalSprite = newSprite;
    }

    public Color GetPixel(int x, int y)
    {
        for (int i = 0; i < Chunks.Length; i++)
        {
            var chunk = Chunks[i];
            if (x >= chunk.X && x < chunk.X + chunk.Sprite.rect.width && y >= chunk.Y && y < chunk.Y + chunk.Sprite.rect.height)
            {
                var chunkSprite = chunk.Sprite;
                var spriteRect = chunkSprite.rect;
                var spriteTexture = chunkSprite.texture;
                var xMin = Mathf.FloorToInt(spriteRect.xMin / spriteTexture.width);
                var yMin = Mathf.FloorToInt(spriteRect.yMin / spriteTexture.height);
                var color = spriteTexture.GetPixel(xMin + x - chunk.X, yMin + y - chunk.Y);
                _colors.Add(color);
            }
        }
        if (_colors.Count == 0)
            return Color.clear;

        var finalColor = _colors[0];
        for (int i = 1; i < _colors.Count; i++)
            if (_colors[i].a != 0)
                finalColor = _colors[i];
        _colors.Clear();
        return finalColor;
    }

    public override bool Equals(object obj)
    {
        if (obj is OptimizedSprite)
        {
            var sprite = obj as OptimizedSprite;
            return sprite &&
                   base.Equals(obj) &&
                   EqualityComparer<OptimizedSpriteChunk[]>.Default.Equals(Chunks, sprite.Chunks) &&
                   Pivot.Equals(sprite.Pivot) &&
                   Width == sprite.Width &&
                   Height == sprite.Height &&
                   EqualityComparer<Sprite>.Default.Equals(OriginalSprite, sprite.OriginalSprite);
        }
        else if (obj is Sprite)
        {
            var sprite = obj as Sprite;
            var spriteTexture = sprite.texture;
            var spriteRect = sprite.rect;
            var xMin = Mathf.FloorToInt(spriteRect.xMin);
            var yMin = Mathf.FloorToInt(spriteRect.yMin);
            for (int x = 0; x < sprite.rect.width; x++)
            {
                for (int y = 0; y < sprite.rect.height; y++)
                {
                    var spritePixel = sprite.texture.GetPixel(xMin + x, yMin + y);
                    var optimizedSpritePixel = GetPixel(x, y);
                    if (spritePixel.a == 0 && optimizedSpritePixel.a == 0)
                        continue;
                    if (spritePixel == optimizedSpritePixel)
                        continue;
                    return false;
                }
            }
            return true;
        }
        return false;
    }
}
