using UnityEngine;

public static class GpuUtilities
{
    public static (uint x, uint y, uint z) GetKernelThreadGroupSizes(this ComputeShader shader, int kernelIndex)
    {
        shader.GetKernelThreadGroupSizes(kernelIndex, out uint x, out uint y, out uint z);
        return (x, y, z);
    }

    public static GpuSpriteDataChunk[] PrepareChunksForGpu(this MyColor[][][] sprites, uint x, uint y, uint z)
    {
        var result = new GpuSpriteDataChunk[sprites.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = new GpuSpriteDataChunk(sprites[i], x, y, z);
        return result;
    }

    public static Texture2D ToTexture2D(this MyColor[][] colors)
    {
        var width = colors.Length;
        var height = colors[0].Length;
        var result = new Texture2D(width, height, TextureFormat.ARGB32, false, false);
        var unityColors = colors.ToUnityColors();
        result.SetPixels(unityColors);
        result.Apply();
        return result;
    }

    public static Color[] ToUnityColors(this MyColor[][] colors)
    {
        var width = colors.Length;
        var height = colors[0].Length;
        var result = new Color[width * height];
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                result[x + y * width] = colors[x][y].ToUnityColor();
        return result;
    }

    public static Color ToUnityColor(this MyColor color) => new Color((float)color.R / byte.MaxValue, (float)color.G / byte.MaxValue, (float)color.B / byte.MaxValue, (float)color.A / byte.MaxValue);
}
