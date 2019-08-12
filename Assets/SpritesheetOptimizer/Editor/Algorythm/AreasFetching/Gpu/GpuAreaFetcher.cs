using System.Collections.Concurrent;
using System.Collections.Generic;
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

        throw new System.NotImplementedException();
    }

    private ComputeBuffer getSpritesBuffer(MyColor[][][] sprites)
    {
        var bufferSize = sprites.Length * sprites[0].Length * sprites[0][0].Length;
        var result = new ComputeBuffer(bufferSize, 32);
        result.SetData(sprites);

        return result;
    }
}
