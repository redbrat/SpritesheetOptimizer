using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

[CreateAssetMenu(fileName = nameof(GetSpritesBySprite), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.FunctionsName + "/" + nameof(GetSpritesBySprite), order = 0)]
public class GetSpritesBySprite : FunctionBase1<Sprite, Sprite[]>
{
    public override Sprite[] Invoke(Sprite sprite)
    {
        var texture = sprite.texture;
        var path = AssetDatabase.GetAssetPath(sprite);
        var allAssetsAtPath = AssetDatabase.LoadAllAssetsAtPath(path);
        var allSptitesAtPath = allAssetsAtPath.OfType<Sprite>().ToArray();
        var ti = AssetImporter.GetAtPath(path) as TextureImporter;
        var fullPath = $"{Application.dataPath.Substring(0, Application.dataPath.Length - "Assets".Length)}{path}";
        //Debug.LogError($"path = {fullPath}");
        texture = new Texture2D(1, 1, TextureFormat.ARGB32, false);
        texture.filterMode = FilterMode.Point;
        texture.LoadImage(File.ReadAllBytes(fullPath));
        //return null;
        ti.isReadable = true;
        ti.SaveAndReimport();

        var spritesCount = ti.spritesheet.Length;

        if (spritesCount == 0) //Если 0, то тип не multiple
        {
            var result = new Sprite[1];
            result[0] = sprite;
            return result;
        }
        else
        {
            var result = new Sprite[spritesCount];
            for (int i = 0; i < spritesCount; i++)
            {
                var currentSprite = ti.spritesheet[i];
                result[i] = allSptitesAtPath.Where(s => s.name == currentSprite.name).First();
            }
            return result;
        }
    }
}
