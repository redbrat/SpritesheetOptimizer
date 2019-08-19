using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

public class GpuAreaFetcher : IAreaFetcher
{
    private readonly ComputeShader _computeShader;

    public GpuAreaFetcher()
    {
        _computeShader = Resources.Load<ComputeShader>("SpritesheetOptimizer/Shaders/AreasFetchingShader");
    }

    Task<ConcurrentDictionary<int, MyArea>> IAreaFetcher.FetchAreas(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var algorythmKernel = _computeShader.FindKernel("CSMain");

        var (groupSizeX, groupSizeY, groupSizeZ) = _computeShader.GetKernelThreadGroupSizes(algorythmKernel);
        var gpuDataChunks = sprites.PrepareChunksForGpu(groupSizeX, groupSizeY, groupSizeZ);
        for (int i = 0; i < gpuDataChunks.Length; i++)
        {
            var chunk = gpuDataChunks[i];
            _computeShader.SetInt("MultipliedValue", chunk.MultipliedPart);
            _computeShader.SetInt("SpriteWidth", chunk.SpriteWidth);
            _computeShader.SetInt("SpriteHeight", chunk.SpriteHeight);
            _computeShader.SetInt("PreferredAreaWidth", 27);
            _computeShader.SetInt("PreferredAreaHeight", 27);
            _computeShader.SetBuffer(algorythmKernel, "SpriteBuffer", chunk.SpriteBuffer);
            _computeShader.SetBuffer(algorythmKernel, "ResultBuffer", chunk.ResultBuffer);
            _computeShader.Dispatch(algorythmKernel, chunk.GroupsCountX, chunk.GroupsCountY, chunk.GroupsCountZ);
            chunk.FetchResults();
            chunk.Dispose();

            var textureBefore = chunk.OriginalSprite.ToTexture2D();
            var textureAfter = chunk.Result.ToTexture2D();
            File.WriteAllBytes($@"C:\results\{i.ToString()}-before.png", textureBefore.EncodeToPNG());
            File.WriteAllBytes($@"C:\results\{i.ToString()}-after.png", textureAfter.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(textureBefore);
            UnityEngine.Object.DestroyImmediate(textureAfter);
        }

        //Здесь что-то делаем с этими данными

        throw new System.NotImplementedException();
    }
    
    private ComputeBuffer getSpritesBuffer(MyColor[][][] sprites)
    {
        var bufferSize = sprites.Length * sprites[0].Length * sprites[0][0].Length;
        var result = new ComputeBuffer(bufferSize, 8);
        result.SetData(sprites);

        return result;
    }
}
