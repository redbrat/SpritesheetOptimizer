using UnityEditor;
using UnityEngine;

[CreateAssetMenu(fileName = nameof(WriteAndImportSprite), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.FunctionsName + "/" + nameof(WriteAndImportSprite), order = 0)]
public class WriteAndImportSprite : FunctionBase3<byte[], DefaultAsset, string, Sprite>
{
    [SerializeField]
    private GetPathByFolderAndName _getPathByFolderAndName;
    [SerializeField]
    private WriteAndImportBytes _writeAndImportBytes;

    public override Sprite Invoke(byte[] bytes, DefaultAsset folder, string fileName)
    {
        var path = _getPathByFolderAndName.Invoke(folder, fileName);
        _writeAndImportBytes.Invoke(bytes, path);

        var ti = AssetImporter.GetAtPath(path) as TextureImporter;
        ti.textureType = TextureImporterType.Sprite;
        ti.filterMode = FilterMode.Point;
        ti.alphaIsTransparency = true;
        ti.mipmapEnabled = false;
        ti.spriteImportMode = SpriteImportMode.Single;
        ti.spritePivot = Vector2.down + Vector2.right;
        ti.isReadable = true;
        ti.crunchedCompression = false;
        ti.textureCompression = TextureImporterCompression.Uncompressed;

        var texSettings = new TextureImporterSettings();
        ti.ReadTextureSettings(texSettings);
        texSettings.spriteAlignment = (int)SpriteAlignment.BottomLeft;
        ti.SetTextureSettings(texSettings);

        AssetDatabase.ImportAsset(path);
        return AssetDatabase.LoadAssetAtPath<Sprite>(path);
    }
}
