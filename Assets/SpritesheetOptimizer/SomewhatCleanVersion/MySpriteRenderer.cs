using UnityEngine;

public class MySpriteRenderer : MonoBehaviour
{
    [SerializeField]
    private OptimizedSprite _optimizedSprite;

    [SerializeField]
    private SpriteRenderer[] _spriteRenderers;

    private void Start()
    {
        _spriteRenderers = new SpriteRenderer[_optimizedSprite.Chunks.Length];
        for (int i = 0; i < _optimizedSprite.Chunks.Length; i++)
        {
            var currentChunk = _optimizedSprite.Chunks[i];
            var newGo = new GameObject($"SpriteRendererHolder_{i}");
            newGo.transform.SetParent(transform);
            newGo.transform.localRotation = default;
            newGo.transform.localScale = Vector3.one;
            newGo.transform.localPosition = new Vector3(currentChunk.X / 100f, currentChunk.Y / 100f);
            _spriteRenderers[i] = newGo.AddComponent<SpriteRenderer>();
            _spriteRenderers[i].sprite = currentChunk.Sprite;
        }
    }
}
