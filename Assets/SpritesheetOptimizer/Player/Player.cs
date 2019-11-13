using UnityEngine;

public class Player : MonoBehaviour
{
    [SerializeField]
    private Texture2D _texture;
    [SerializeField]
    private TextAsset _blob;

    [SerializeField]
    private SpriteRenderer[] _spriteRenderers;
    [SerializeField]
    private Sprite[] _sprites;

    private BlobPlayer _blobPlayer;

    [SerializeField]
    private int _currentFrame;

    private void Start()
    {
        Initialize();
    }

    [ContextMenu("Initialize")]
    public void Initialize()
    {
        _blobPlayer = new BlobPlayer(_blob.bytes);
        if (_spriteRenderers == default || _spriteRenderers.Length != _blobPlayer.ChunksCount)
        {
            deinitializeRenderers();
            initializeRenderers();
        }
        if (_sprites == default || _sprites.Length != _blobPlayer.AtlasLength)
        {
            deinitializeAtlas();
            initializeAtlas();
        }
    }

    private void initializeAtlas()
    {
        _sprites = new Sprite[_blobPlayer.AtlasLength];
        for (int i = 0; i < _sprites.Length; i++)
        {
            var rect = _blobPlayer.GetAtlasEntry(i);
            _sprites[i] = Sprite.Create(_texture, new Rect(rect.x, rect.y, rect.width, rect.height), Vector2.zero, 100);
        }
    }

    private void deinitializeAtlas()
    {
        if (_sprites == default)
            return;
        for (int i = 0; i < _sprites.Length; i++)
            DestroyImmediate(_sprites[i]);
        _sprites = default;
    }

    private void initializeRenderers()
    {
        _spriteRenderers = new SpriteRenderer[_blobPlayer.ChunksCount];
        for (int i = 0; i < _spriteRenderers.Length; i++)
        {
            var newRendererGo = new GameObject($"Chunk_{i}");
            newRendererGo.transform.SetParent(transform);
            newRendererGo.transform.localPosition = Vector3.zero;
            newRendererGo.transform.localRotation = Quaternion.identity;
            newRendererGo.transform.localScale = Vector3.one;
            _spriteRenderers[i] = newRendererGo.AddComponent<SpriteRenderer>();
        }
    }

    private void deinitializeRenderers()
    {
        if (_spriteRenderers == default)
            return;
        for (int i = 0; i < _spriteRenderers.Length; i++)
            Destroy(_spriteRenderers[i].gameObject);
        _spriteRenderers = default;
    }

    public void Deinitialize()
    {
        _blobPlayer = default;
    }

    private void Update()
    {
        if (_blobPlayer == default)
            return;

        var chunks = _blobPlayer.GetFrame(_currentFrame % _blobPlayer.FramesCount);
        for (int i = 0; i < chunks.Length; i++)
        {
            var chunk = chunks[i];
            var spriteRenderer = _spriteRenderers[i];
            spriteRenderer.sprite = chunk == default ? default : _sprites[chunk.AtlasIndex];
            if (chunk != default)
                spriteRenderer.transform.localPosition = new Vector3(chunk.X / 100f, chunk.Y / 100f, 0f);
        }
    }
}
