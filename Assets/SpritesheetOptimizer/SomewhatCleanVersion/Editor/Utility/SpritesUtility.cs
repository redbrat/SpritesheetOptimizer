using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class SpritesUtility
{
    public static (MyColor[][][] colors, byte[][][][] bytes, Sprite[] sprites) GetAllSpriteBytesAndColors(this Sprite sprite)
    {
        var (bytes, sprites) = sprite.GetAllSpriteBytes();
        var colors = new MyColor[bytes.Length][][];
        for (int i = 0; i < colors.Length; i++)
        {
            colors[i] = new MyColor[bytes[i].Length][];
            for (int x = 0; x < colors[i].Length; x++)
            {
                colors[i][x] = new MyColor[bytes[i][x].Length];
                for (int y = 0; y < colors[i][x].Length; y++)
                {
                    var currentBytes = bytes[i][x][y];
                    colors[i][x][y] = new MyColor(currentBytes[0], currentBytes[1], currentBytes[2], currentBytes[3]);
                }
            }
        }
        return (colors, bytes, sprites);
    }

    public static (byte[][][][] bytes, Sprite[] sprites) GetAllSpriteBytes(this Sprite sprite)
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
        var bytes = default(byte[][][][]);
        var sprites = default(Sprite[]);
        if (spritesCount == 0) //If there're no items in spritesheet - it means there is a single sprite in asset.
        {
            bytes = new byte[1][][][];
            sprites = new Sprite[1];
            sprites[0] = sprite;

            var tex = sprite.texture;
            var currentBytes = new byte[tex.width][][];
            for (int x = 0; x < tex.width; x++)
            {
                currentBytes[x] = new byte[tex.height][];
                for (int y = 0; y < tex.height; y++)
                {
                    var color = texture.GetPixel(x, y);
                    currentBytes[x][y] = new byte[4];
                    currentBytes[x][y][0] = Convert.ToByte(Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue));
                    currentBytes[x][y][1] = Convert.ToByte(Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue));
                    currentBytes[x][y][2] = Convert.ToByte(Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue));
                    currentBytes[x][y][3] = Convert.ToByte(Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue));
                }
            }
            bytes[0] = currentBytes;
        }
        else
        {
            bytes = new byte[spritesCount][][][];
            sprites = new Sprite[spritesCount];

            for (int i = 0; i < spritesCount; i++)
            {
                var currentSprite = ti.spritesheet[i];
                sprites[i] = allSptitesAtPath.Where(s => s.name == currentSprite.name).First();

                var xOrigin = Mathf.FloorToInt(sprites[i].rect.x);
                var yOrigin = Mathf.CeilToInt(sprites[i].rect.y);
                var width = Mathf.CeilToInt(sprites[i].rect.width);
                var height = Mathf.CeilToInt(sprites[i].rect.height);
                var currentBytes = new byte[width][][];

                for (int x = 0; x < width; x++)
                {
                    currentBytes[x] = new byte[height][];
                    for (int y = 0; y < height; y++)
                    {
                        var color = texture.GetPixel(xOrigin + x, yOrigin + y);
                        var r = (byte)Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue);
                        var g = (byte)Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue);
                        var b = (byte)Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue);
                        var a = (byte)Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue);
                        currentBytes[x][y] = new byte[4] { r, g, b, a };
                    }
                }
                bytes[i] = currentBytes;
            }
        }

        return (bytes, sprites);
    }
}
