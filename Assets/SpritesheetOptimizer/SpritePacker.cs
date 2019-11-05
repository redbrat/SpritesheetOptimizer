using System.IO;
using UnityEngine;

public class SpritePacker : MonoBehaviour
{
    [SerializeField]
    private string _path = @"P:\U\Some2DGame\Assets\atlas.png";

    [SerializeField]
    private Texture2D[] _textures;

    [ContextMenu("Pack")]
    private void pack()
    {
        var atlas = new Texture2D(1, 1, TextureFormat.RGBA32, false, false);
        atlas.filterMode = FilterMode.Point;
        atlas.PackTextures(_textures, 0);
        File.WriteAllBytes(_path, atlas.EncodeToPNG());
    }
}
