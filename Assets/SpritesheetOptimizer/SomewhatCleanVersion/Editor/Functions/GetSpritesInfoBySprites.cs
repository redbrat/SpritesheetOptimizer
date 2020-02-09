using UnityEngine;

[CreateAssetMenu(fileName = nameof(GetSpritesInfoBySprites), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.FunctionsName + "/" + nameof(GetSpritesInfoBySprites), order = 0)]
public class GetSpritesInfoBySprites : FunctionBase1<Sprite[], SpritesInfo>
{
    public override SpritesInfo Invoke(Sprite[] sprites)
    {
        var bytes = new byte[sprites.Length][][][];

        for (int i = 0; i < sprites.Length; i++)
        {
            var sprite = sprites[i];

            var xOrigin = Mathf.FloorToInt(sprite.rect.x);
            var yOrigin = Mathf.CeilToInt(sprite.rect.y);
            var width = Mathf.CeilToInt(sprite.rect.width);
            var height = Mathf.CeilToInt(sprite.rect.height);
            var currentBytes = new byte[width][][];

            for (int x = 0; x < width; x++)
            {
                currentBytes[x] = new byte[height][];
                for (int y = 0; y < height; y++)
                {
                    var color = sprite.texture.GetPixel(xOrigin + x, yOrigin + y);
                    var r = (byte)Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue);
                    var g = (byte)Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue);
                    var b = (byte)Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue);
                    var a = (byte)Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue);
                    currentBytes[x][y] = new byte[4] { r, g, b, a };
                }
            }
            bytes[i] = currentBytes;
        }

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

        return new SpritesInfo(colors, bytes, sprites);
    }
}
