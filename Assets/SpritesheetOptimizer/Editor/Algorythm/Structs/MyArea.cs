﻿using System.Collections.Concurrent;
using System.Text;
using UnityEngine;

public class MyArea
{
    public readonly MyColor[] _colors;

    public readonly MyVector2 Dimensions;

    public readonly int OpaquePixelsCount;

    private readonly int _hash;

    public readonly ConcurrentDictionary<int, MyAreaCoordinates> Correlations;

    public readonly ConcurrentBag<MyAreaCoordinates> CorrelationsBag;

    public readonly long Score;

    public readonly int SpriteIndex;

    public readonly MyRect SpriteRect;

    public bool Selected;

    //public MyArea(MyVector2 dimensions, params MyColor[] colors)
    //{
    //    _colors = colors;
    //    Dimensions = dimensions;
    //    Correlations = new ConcurrentDictionary<int, MyAreaCoordinates>();

    //    OpaquePixelsCount = 0;
    //    _hash = 0;
    //    for (int i = 0; i < _colors.Length; i++)
    //    {
    //        if (_colors[i].A > 0f)
    //            OpaquePixelsCount++;
    //        _hash += (i + 1) * _colors[i].GetHashCode() * short.MaxValue;
    //    }
    //    _hash += dimensions.GetHashCode();

    //    Score = (long)(Mathf.Pow(OpaquePixelsCount, 3f) / Dimensions.Square);
    //}

    public readonly string UniqueString;

    public MyArea(MyVector2 dimensions, int spriteIndex, MyRect spriteRect, string uniqueString, params MyColor[] colors)
    {
        _colors = colors;
        Dimensions = dimensions;
        Correlations = new ConcurrentDictionary<int, MyAreaCoordinates>();
        CorrelationsBag = new ConcurrentBag<MyAreaCoordinates>();

        OpaquePixelsCount = 0;
        for (int i = 0; i < _colors.Length; i++)
            if (_colors[i].A > 0)
                OpaquePixelsCount++;

        Score = (long)(Mathf.Pow(OpaquePixelsCount, 3f) / Dimensions.Square);

        SpriteIndex = spriteIndex;
        SpriteRect = spriteRect;

        UniqueString = uniqueString;
        _hash = uniqueString.GetHashCode();
    }


    public MyArea(MyVector2 dimensions, int spriteIndex, MyRect spriteRect, params MyColor[] colors)
    {
        _colors = colors;
        Dimensions = dimensions;
        Correlations = new ConcurrentDictionary<int, MyAreaCoordinates>();
        CorrelationsBag = new ConcurrentBag<MyAreaCoordinates>();

        OpaquePixelsCount = 0;
        _hash = 0;
        for (int i = 0; i < _colors.Length; i++)
        {
            if (_colors[i].A > 0f)
                OpaquePixelsCount++;
            _hash += (i + 1) * _colors[i].GetHashCode() * short.MaxValue;
        }
        _hash += dimensions.GetHashCode();
        _hash += spriteIndex.GetHashCode();
        _hash += spriteRect.GetHashCode();
        _hash += OpaquePixelsCount.GetHashCode();

        Score = (long)(Mathf.Pow(OpaquePixelsCount, 3f) / Dimensions.Square);

        SpriteIndex = spriteIndex;
        SpriteRect = spriteRect;

        UniqueString = _hash.ToString();
    }

    public override int GetHashCode() => _hash;

    public static bool ContainsOpaquePixels(MyColor[][] sprite, int x, int y, MyVector2 dimensions)
    {
        for (int xx = 0; xx < dimensions.X; xx++)
            for (int yy = 0; yy < dimensions.Y; yy++)
                if (sprite[x + xx][y + yy].A > 0)
                    return true;
        return false;
    }

    public static MyArea CreateFromSprite(MyColor[][] sprite, int spriteIndex, int x, int y, MyVector2 dimensions)
    {
        var sb = new StringBuilder();
        sb.Clear();
        var colors = new MyColor[dimensions.Square];
        for (int xx = 0; xx < dimensions.X; xx++)
        {
            sb.Append('_');
            sb.Append(xx);
            for (int yy = 0; yy < dimensions.Y; yy++)
            {
                var colorsIndex = xx + yy * dimensions.X;
                var spriteX = x + xx;
                var spriteY = y + yy;

                var pixel = sprite[spriteX][spriteY];

                if (colorsIndex > colors.Length - 1)
                    Debug.LogError($"colors[colorsIndex] is out of range! colors.Length = {colors.Length}, colorsIndex = {colorsIndex}");
                if (spriteX > sprite.Length - 1)
                    Debug.LogError($"sprite[spriteX] is out of range! sprite.Length = {sprite.Length}, spriteX = {spriteX}");
                if (spriteY > sprite[spriteX].Length - 1)
                    Debug.LogError($"sprite[spriteX][spriteY] is out of range! sprite[spriteX].Length = {sprite[spriteX].Length}, spriteY = {spriteY}");
                colors[xx + yy * dimensions.X] = pixel;

                sb.Append('-');
                sb.Append(xx);
                sb.Append('.');
                sb.Append(yy);
                sb.Append('&');
                sb.Append(pixel.R);
                sb.Append('?');
                sb.Append(pixel.G);
                sb.Append('#');
                sb.Append(pixel.B);
                sb.Append('!');
                sb.Append(pixel.A);
            }
        }
        //for (int yy = 0; yy < dimensions.Y; yy++)
        //{
        //    var colorsIndex = xx + yy * dimensions.X;
        //    var spriteX = x + xx;
        //    var spriteY = y + yy;
        //    if (colorsIndex > colors.Length - 1)
        //        Debug.LogError($"colors[colorsIndex] is out of range! colors.Length = {colors.Length}, colorsIndex = {colorsIndex}");
        //    if (spriteX > sprite.Length - 1)
        //        Debug.LogError($"sprite[spriteX] is out of range! sprite.Length = {sprite.Length}, spriteX = {spriteX}");
        //    if (spriteY > sprite[spriteX].Length - 1)
        //        Debug.LogError($"sprite[spriteX][spriteY] is out of range! sprite[spriteX].Length = {sprite[spriteX].Length}, spriteY = {spriteY}");
        //    colors[xx + yy * dimensions.X] = sprite[x + xx][y + yy];
        //}

        return new MyArea(dimensions, spriteIndex, new MyRect(x, y, dimensions.X, dimensions.Y), sb.ToString(), colors);
    }

    //public static MyArea CreateFromSprite(MyColor[][] sprite, int x, int y, MyVector2 dimensions)
    //{
    //    var colors = new MyColor[dimensions.Square];
    //    for (int xx = 0; xx < dimensions.X; xx++)
    //        for (int yy = 0; yy < dimensions.Y; yy++)
    //        {
    //            var colorsIndex = xx + yy * dimensions.X;
    //            var spriteX = x + xx;
    //            var spriteY = y + yy;
    //            if (colorsIndex > colors.Length - 1)
    //                Debug.LogError($"colors[colorsIndex] is out of range! colors.Length = {colors.Length}, colorsIndex = {colorsIndex}");
    //            if (spriteX > sprite.Length - 1)
    //                Debug.LogError($"sprite[spriteX] is out of range! sprite.Length = {sprite.Length}, spriteX = {spriteX}");
    //            if (spriteY > sprite[spriteX].Length - 1)
    //                Debug.LogError($"sprite[spriteX][spriteY] is out of range! sprite[spriteX].Length = {sprite[spriteX].Length}, spriteY = {spriteY}");
    //            colors[xx + yy * dimensions.X] = sprite[x + xx][y + yy];
    //        }
    //    return new MyArea(dimensions, colors);
    //}

    public static void EraseAreaFromSprite(MyColor[][] sprite, int x, int y, MyVector2 dimensions)
    {
        for (int xx = 0; xx < dimensions.X; xx++)
            for (int yy = 0; yy < dimensions.Y; yy++)
                sprite[x + xx][y + yy] = new MyColor(byte.MinValue, byte.MinValue, byte.MinValue, byte.MinValue);
    }

    public static void EraseUpdateEmptinessMap(MyColor[][] sprite, bool[][] spritesMapOfEmptiness, int x, int y, MyVector2 erasedAreaDimensions, MyVector2 updatedAreaDimensions)
    {
        for (int xx = 0; xx < erasedAreaDimensions.X; xx++)
            for (int yy = 0; yy < erasedAreaDimensions.Y; yy++)
            {
                var spriteXCoord = x + xx;
                var spriteYCoord = y + yy;
                if (spriteXCoord + updatedAreaDimensions.X >= sprite.Length || spriteYCoord + updatedAreaDimensions.Y >= sprite[spriteXCoord].Length)
                    continue;
                spritesMapOfEmptiness[spriteXCoord][spriteYCoord] = !ContainsOpaquePixels(sprite, spriteXCoord, spriteYCoord, updatedAreaDimensions);
            }
    }
}