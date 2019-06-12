using System;

public class DefaultAreaEnumerator : IAreaEnumerator
{
    private readonly MyColor[][][] _sprites;

    public DefaultAreaEnumerator(MyColor[][][] sprites)
    {
        _sprites = sprites;
    }

    public void EnumerateThroughSprite(MyVector2 areaSizing, int spriteIndex, Action<MyColor[][], int, int> action)
    {
        var sprite = _sprites[spriteIndex];
        for (int x = 0; x < sprite.Length; x++)
            for (int y = 0; y < sprite[x].Length; y++)
                action(sprite, x, y);
    }

    public void Enumerate(MyVector2 areaSizing, Action<MyColor[][], int, int> action)
    {
        for (int i = 0; i < _sprites.Length; i++)
            for (int x = 0; x < _sprites[i].Length; x++)
                for (int y = 0; y < _sprites[i][x].Length; y++)
                    action(_sprites[i], x, y);
    }

    public void EnumerateCopy(MyVector2 areaDimensions, Action<MyColor[][], int, int> action)
    {
        var copy = CopyArrayOf(_sprites);
        for (int i = 0; i < copy.Length; i++)
            for (int x = 0; x < copy[i].Length; x++)
                for (int y = 0; y < copy[i][x].Length; y++)
                    action(copy[i], x, y);
    }

    private T[][][] CopyArrayOf<T>(T[][][] source)
    {
        var len = source.Length;
        var dest = new T[len][][];

        for (int i = 0; i < len; i++)
        {
            var len2 = source[i].Length;
            dest[i] = new T[len2][];

            for (int j = 0; j < len2; j++)
            {
                var len3 = source[i][j].Length;
                dest[i][j] = new T[len3];

                Array.Copy(source[i][j], dest[i][j], len3);
            }
        }

        return dest;
    }
}
