using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

public class GpuAreaFetcher : IAreaFetcher
{
    private readonly ComputeShader _computeShader;
    private readonly ComputeShader _computeShaderLegacy;

    private readonly Dictionary<MyVector2, bool[][][]> _mapOfEmptiness;
    private bool _mapOfEmptinessInitialized;

    public GpuAreaFetcher()
    {
        _computeShader = Resources.Load<ComputeShader>("SpritesheetOptimizer/Shaders/AreasFetchingShader");
        _computeShaderLegacy = Resources.Load<ComputeShader>("SpritesheetOptimizer/Shaders/AreasFetchingShaderLegacy");

        _mapOfEmptiness = new Dictionary<MyVector2, bool[][][]>();
    }

    /*
     * Ок, мы хотим создать задания - пройтись каждой уникальной областью по всем спрайтам. Чтобы получить уникальные области - возможно тоже придется 
     * выкручиваться с помощью шейдеров. Но пока можно и тупо пройтись процессором. 
     * 
     * Значит, что нам надо сделать - пройтись по всем спрайтам, составить список всех уникальных областей всех размеров во всех спрайтах. Затем надо сложить 
     * инжу о них в буффер так, чтобы на каждом пикселе находилась инфа о том, где находится эта область, т.е. x, y, width, height вместо r, g, b, a. Дальше 
     * шейдер идут туда забираер эту область и проходится с ней по всем спрайтам, считает счет и пишет его в соответствующую клетку резултирующего буффера.
     */

    Task<ConcurrentDictionary<int, MyArea>> IAreaFetcher.FetchAreas(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        return Task.Run(() =>
        {
            if (!_mapOfEmptinessInitialized)
            {
                Debug.LogError($"map estimation start");
                foreach (var area in areaSizings)
                {
                    var currentMapOfEmptiness = new bool[sprites.Length][][];
                    for (int j = 0; j < sprites.Length; j++)
                    {
                        var sprite = sprites[j];
                        var spriteMapOfEmptiness = new bool[sprite.Length - area.X][];
                        for (int x = 0; x < sprite.Length - area.X; x++) 
                        {
                            spriteMapOfEmptiness[x] = new bool[sprite[x].Length - area.Y];
                            for (int y = 0; y < sprite[x].Length - area.Y; y++)
                                spriteMapOfEmptiness[x][y] = !MyArea.ContainsOpaquePixels(sprite, x, y, area);
                        }
                        currentMapOfEmptiness[j] = spriteMapOfEmptiness;
                    }
                    _mapOfEmptiness.Add(area, currentMapOfEmptiness);
                }
                _mapOfEmptinessInitialized = true;
                Debug.LogError($"map estimation end");
            }

            var result = new ConcurrentDictionary<int, MyArea>(); //Тут мы храним все уникальные области по их хешам
            var overallOpsCount = areaSizings.Count() * sprites.Length;
            var areasArray = areaSizings.ToArray();

            progressReport.OperationsCount = overallOpsCount;

            try
            {
                Parallel.For(0, overallOpsCount, (int index, ParallelLoopState state) =>
                {
                    if (state.IsExceptional)
                        Debug.Log("Exception!");
                    var areaVariantIndex = Mathf.FloorToInt(index / sprites.Length);
                    var spriteIndex = index - areaVariantIndex * sprites.Length;
                    var targetArea = areasArray[areaVariantIndex];
                    var mapOfEmptinessForAreaAndSprite = _mapOfEmptiness[targetArea][spriteIndex];
                    getUniqueAreas(targetArea, spriteIndex, sprites[spriteIndex], result, mapOfEmptinessForAreaAndSprite, progressReport);
                });
            }
            catch (AggregateException ae)
            {
                Debug.Log("catch");
                ae.Handle((inner) =>
                {
                    Debug.Log(inner.Message);
                    return true;
                });
            }

            Debug.Log($"unique areas count = {result.Count}");
            return result;
        });

        //var algorythmKernel = _computeShader.FindKernel("CSMain");

        //var (groupSizeX, groupSizeY, groupSizeZ) = _computeShader.GetKernelThreadGroupSizes(algorythmKernel);
        //var gpuDataChunks = sprites.PrepareChunksForGpu(groupSizeX, groupSizeY, groupSizeZ);
        //for (int i = 0; i < gpuDataChunks.Length; i++)
        //{
        //    var chunk = gpuDataChunks[i];
        //    _computeShader.SetInt("MultipliedValue", chunk.MultipliedPart);
        //    _computeShader.SetInt("SpriteWidth", chunk.SpriteWidth);
        //    _computeShader.SetInt("SpriteHeight", chunk.SpriteHeight);
        //    _computeShader.SetInt("PreferredAreaWidth", 27);
        //    _computeShader.SetInt("PreferredAreaHeight", 27);
        //    _computeShader.SetBuffer(algorythmKernel, "SpriteBuffer", chunk.SpriteBuffer);
        //    _computeShader.SetBuffer(algorythmKernel, "ResultBuffer", chunk.ResultBuffer);
        //    _computeShader.Dispatch(algorythmKernel, chunk.GroupsCountX, chunk.GroupsCountY, chunk.GroupsCountZ);
        //    chunk.FetchResults();
        //    chunk.Dispose();

        //    var textureBefore = chunk.OriginalSprite.ToTexture2D();
        //    var textureAfter = chunk.Result.ToTexture2D();
        //    File.WriteAllBytes($@"C:\results\{i.ToString()}-before.png", textureBefore.EncodeToPNG());
        //    File.WriteAllBytes($@"C:\results\{i.ToString()}-after.png", textureAfter.EncodeToPNG());
        //    UnityEngine.Object.DestroyImmediate(textureBefore);
        //    UnityEngine.Object.DestroyImmediate(textureAfter);
        //}

        //Здесь что-то делаем с этими данными

        throw new System.NotImplementedException();
    }

    private static void getUniqueAreas(MyVector2 areaResolution, int spriteIndex, MyColor[][] sprite, ConcurrentDictionary<int, MyArea> areas, bool[][] mapOfEmptiness, ProgressReport progressReport)
    {
        var areaSquare = areaResolution.X * areaResolution.Y;
        for (int x = 0; x < sprite.Length - areaResolution.X; x++)
        {
            for (int y = 0; y < sprite[x].Length - areaResolution.Y; y++)
            {
                if (mapOfEmptiness[x][y])
                    continue;
                var area = MyArea.CreateFromSprite(sprite, spriteIndex, x, y, areaResolution);
                if (area.Score == 0)
                    continue;
                var hash = area.GetHashCode();
                areas.TryAdd(hash, area);
            }
        }
        progressReport.OperationsDone++;
    }

    Task<ConcurrentDictionary<int, MyArea>> FetchAreasLegacy(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var algorythmKernel = _computeShaderLegacy.FindKernel("CSMain");

        var (groupSizeX, groupSizeY, groupSizeZ) = _computeShaderLegacy.GetKernelThreadGroupSizes(algorythmKernel);
        var gpuDataChunks = sprites.PrepareChunksForGpuLegacy(groupSizeX, groupSizeY, groupSizeZ);
        for (int i = 0; i < gpuDataChunks.Length; i++)
        {
            var chunk = gpuDataChunks[i];
            _computeShaderLegacy.SetInt("MultipliedValue", chunk.MultipliedPart);
            _computeShaderLegacy.SetInt("SpriteWidth", chunk.SpriteWidth);
            _computeShaderLegacy.SetInt("SpriteHeight", chunk.SpriteHeight);
            _computeShaderLegacy.SetInt("PreferredAreaWidth", 27);
            _computeShaderLegacy.SetInt("PreferredAreaHeight", 27);
            _computeShaderLegacy.SetBuffer(algorythmKernel, "SpriteBuffer", chunk.SpriteBuffer);
            _computeShaderLegacy.SetBuffer(algorythmKernel, "ResultBuffer", chunk.ResultBuffer);
            _computeShaderLegacy.Dispatch(algorythmKernel, chunk.GroupsCountX, chunk.GroupsCountY, chunk.GroupsCountZ);
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
