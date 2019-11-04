using UnityEngine;

public class OpaquePixelsCounter : MonoBehaviour
{
    [SerializeField]
    private int _count;

    [ContextMenu("Count")]
    private void count()
    {
        var allSpriteRenderers = GetComponentsInChildren<SpriteRenderer>();
        var count = 0;
        for (int i = 0; i < allSpriteRenderers.Length; i++)
        {
            var sprite = allSpriteRenderers[i].sprite;
            if (sprite == default)
                continue;
            var texture = sprite.texture;
            for (int x = 0; x < texture.width; x++)
            {
                for (int y = 0; y < texture.height; y++)
                {
                    var pixel = texture.GetPixel(x, y);
                    if (pixel.a != 0f)
                        count++;
                }
            }
        }
        _count = count;
    }
}
