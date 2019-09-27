using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class Algorythm
{
    public readonly ProgressReport OverallProgressReport;
    public readonly ProgressReport OperationProgressReport;

    public int UnprocessedPixels { get; private set; }

    /// <summary>
    /// Сколько раз можно воспользоваться списком областей перед их устареванием
    /// </summary>
    private readonly int _areasFreshmentSpan;

    /// <summary>
    /// Сколько областей могут быть предположительно затронуты удалением одной области с картинки и должны быть обновлены перед переупорядочиванием
    /// </summary>
    private readonly int _areasVolatilityRange;

    private readonly MyColor[][][] _sprites;
    private readonly IList<ISizingsConfigurator> _sizingsConfigurators;
    private readonly IList<IScoreCounter> _scoreCounters;
    private readonly Type _areaEnumeratorType;
    private readonly ComputeMode _computeMode;

    private readonly ComputeShader _computeShader;

    private ConcurrentDictionary<int, MyArea> _allAreas;
    private MapOfEmptiness _mapOfEmptiness;
    private IEnumerable<MyVector2> _areaSizings;
    private IAreaEnumerator _areaEnumerator;
    private IAreaFetcher _areaFetcher;

    private CancellationToken _ct;

    public Algorythm(MyColor[][][] sprites, Type areaEnumeratorType, IList<ISizingsConfigurator> sizingConfigurators, IList<IScoreCounter> scoreCounters, int areasFreshmentSpan, int areasVolatilityRange, ComputeMode computeMode)
    {
        OperationProgressReport = new ProgressReport();
        OverallProgressReport = new ProgressReport();

        _areasFreshmentSpan = areasFreshmentSpan;
        _areasVolatilityRange = areasVolatilityRange;

        _sprites = sprites;
        _areaEnumeratorType = areaEnumeratorType;
        _sizingsConfigurators = sizingConfigurators;
        _scoreCounters = scoreCounters;

        _computeMode = computeMode;

        _computeShader = Resources.Load<ComputeShader>("SpritesheetOptimizer/Shaders/AreasFetchingShader");
    }

    public async Task Initialize(Vector2Int maxAreaSize, CancellationToken ct)
    {
        Debug.LogError($"Initialize 1");
        _ct = ct;
        _areaSizings = getAreaSizings(_sprites, maxAreaSize, ct);

        var areaEnumeratorCtor = _areaEnumeratorType.GetConstructor(new Type[] { typeof(MyColor[][][]) });
        if (areaEnumeratorCtor == null)
            areaEnumeratorCtor = _areaEnumeratorType.GetConstructor(new Type[] { typeof(MyColor[][][]), typeof(IEnumerable<MyVector2>) });
        else
            _areaEnumerator = (IAreaEnumerator)areaEnumeratorCtor.Invoke(new object[] { _sprites });
        Debug.LogError($"Initialize 2");
        if (_areaEnumerator == null && areaEnumeratorCtor == null)
            throw new ArgumentException($"Got AreaEnumerator with unknown set of ctor parameters.");
        else if (_areaEnumerator == null)
            _areaEnumerator = (IAreaEnumerator)areaEnumeratorCtor.Invoke(new object[] { _sprites, _areaSizings });
        Debug.LogError($"Initialize 3");
        UnprocessedPixels = countUprocessedPixels(MyVector2.One, _areaEnumerator);
        Debug.LogError($"Initialize 4");
        //_mapOfEmptiness = new MapOfEmptiness();
        //await _mapOfEmptiness.Initialize(_areaSizings, _sprites, _areaEnumerator);
        Debug.LogError($"Initialize 5");

        switch (_computeMode)
        {
            case ComputeMode.Cpu:
                _areaFetcher = new CpuAreaFetcher(_ct, _mapOfEmptiness);
                break;
            case ComputeMode.Gpu:
                _areaFetcher = new GpuAreaFetcher();
                break;
        }
        Debug.LogError($"Initialize 6");
    }

    #region Initializing

    private IEnumerable<MyVector2> getAreaSizings(MyColor[][][] sprites, Vector2Int maxAreaSize, CancellationToken ct)
    {
        var result = default(IEnumerable<MyVector2>);
        for (int i = 0; i < _sizingsConfigurators.Count && !ct.IsCancellationRequested; i++)
            result = _sizingsConfigurators[i]?.ConfigureSizings(result, sprites.Length, maxAreaSize.x, maxAreaSize.y, ct);
        return result;
    }

    private int countUprocessedPixels(MyVector2 areaSizing, IAreaEnumerator areaEnumerator)
    {
        var result = 0;
        areaEnumerator.Enumerate(areaSizing, (sprite, index, x, y) =>
        {
            if (sprite[x][y].A > 0)
                result++;
        });
        return result;
    }

    #endregion Initializing

    private struct areaStruct
    {
        public int SpriteIndex;
        public int XAndY; //Экономим место, т.к. эти буфферы тянут на сотни мегабайт...
        public int WidthAndHeight;

        public areaStruct(int spriteIndex, int xAndY, int widthAndHeight)
        {
            SpriteIndex = spriteIndex;
            XAndY = xAndY;
            WidthAndHeight = widthAndHeight;
        }
    }

    private struct registryStruct
    {
        public int SpritesDataOffset; //С какой позиции в буффере data начинается данный спрайт
        public int WidthAndHeight;

        public registryStruct(int spritesDataOffset, int widthAndHeight)
        {
            SpritesDataOffset = spritesDataOffset;
            WidthAndHeight = widthAndHeight;
        }
    }

    public async Task<Dictionary<MyArea, List<MyAreaCoordinates>>> Run()
    {
        OverallProgressReport.OperationDescription = "Removing areas from picture";
        OverallProgressReport.OperationsCount = UnprocessedPixels;
        var result = new Dictionary<MyArea, List<MyAreaCoordinates>>();

        Debug.Log($"Составляем список всех уникальных областей...");
        await setAreasAndScores();
        Debug.Log($"Список составили, начали составлять буфер");

        /*
         * Т.к. это GPU мы должны положить этот список в буфер
         * Собсно у нас, самое главное, должен быть регистр - список всех спрайтов, по которым бы будем проходиться, 
         * и из которых будут браться области, т.е. адресное пространство. Собсно, нужно разделить его на регистр и
         * собсно хранилище. Через регистр у нас будут сдвиги по нужным осям. Собсно ось будет всего одна, т.к. размеры
         * текстур совершенно разные.
         */

        //var registryEntrySize = 8;
        //var registry = new byte[_sprites.Length * registryEntrySize];
        //var pixelEntrySize = 4;
        //var dataSize = 0;
        //for (int i = 0; i < _sprites.Length; i++)
        //    dataSize += _sprites[i].Length * _sprites[i][0].Length;
        //var data = new byte[dataSize * pixelEntrySize];

        //var registryOffset = 0;
        //var dataOffset = 0;
        //for (int i = 0; i < _sprites.Length; i++)
        //{
        //    var sprite = _sprites[i];
        //    Buffer.BlockCopy(BitConverter.GetBytes(registryOffset), 0, registry, registryOffset, 4);
        //    var width = (short)sprite.Length;
        //    var height = (short)sprite[0].Length;
        //    Buffer.BlockCopy(BitConverter.GetBytes(width), 0, registry, registryOffset + 2, 2);
        //    Buffer.BlockCopy(BitConverter.GetBytes(height), 0, registry, registryOffset + 4, 2);
        //    registryOffset += registryEntrySize;

        //    for (int x = 0; x < sprite.Length; x++)
        //    {
        //        for (int y = 0; y < sprite[x].Length; y++)
        //        {
        //            data[dataOffset++] = sprite[x][y].R;
        //            data[dataOffset++] = sprite[x][y].G;
        //            data[dataOffset++] = sprite[x][y].B;
        //            data[dataOffset++] = sprite[x][y].A;
        //        }
        //    }
        //}

        //var areaEntrySize = 12;
        //var areas = new byte[areaEntrySize * _allAreas.Count()];
        //var areaOffset = 0;
        //foreach (var kvp in _allAreas)
        //{
        //    var area = kvp.Value;
        //    Buffer.BlockCopy(BitConverter.GetBytes(area.SpriteIndex), 0, areas, areaOffset, 4);
        //    Buffer.BlockCopy(BitConverter.GetBytes((short)area.SpriteRect.X), 0, areas, areaOffset + 4, 2);
        //    Buffer.BlockCopy(BitConverter.GetBytes((short)area.SpriteRect.Y), 0, areas, areaOffset + 6, 2);
        //    Buffer.BlockCopy(BitConverter.GetBytes((short)area.SpriteRect.Width), 0, areas, areaOffset + 8, 2);
        //    Buffer.BlockCopy(BitConverter.GetBytes((short)area.SpriteRect.Height), 0, areas, areaOffset + 10, 2);
        //    areaOffset += areaEntrySize;
        //}

        var dataSize = 0;
        for (int i = 0; i < _sprites.Length; i++)
            dataSize += _sprites[i].Length * _sprites[i][0].Length;
        var data = new int[dataSize];

        var registry = new registryStruct[_sprites.Length];

        var dataOffset = 0;
        for (int i = 0; i < _sprites.Length; i++)
        {
            var sprite = _sprites[i];
            var width = sprite.Length;
            var height = sprite[0].Length;

            registry[i] = new registryStruct(dataOffset, width << 16 | height);

            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                    data[dataOffset++] = sprite[x][y].R << 24 | sprite[x][y].G << 16 | sprite[x][y].B << 8 | sprite[x][y].A;
        }

        var areasList = _allAreas.Select(kvp => kvp.Value).ToList();
        var areas = new areaStruct[areasList.Count];
        for (int i = 0; i < areasList.Count; i++)
        {
            var area = areasList[i];
            areas[i] = new areaStruct(area.SpriteIndex, area.SpriteRect.X << 16 | area.SpriteRect.Y, area.SpriteRect.Width << 16 | area.SpriteRect.Height);
        }

        var algorythmKernel = _computeShader.FindKernel("CSMain");
        var (groupSizeX, groupSizeY, groupSizeZ) = _computeShader.GetKernelThreadGroupSizes(algorythmKernel);

        var areasBuffer = new ComputeBuffer(areasList.Count, 12);
        areasBuffer.SetData(areas);

        var registryBuffer = new ComputeBuffer(_sprites.Length, 8);
        registryBuffer.SetData(registry);

        var dataBuffer = new ComputeBuffer(dataSize, 4);
        dataBuffer.SetData(data);

        var resultBuffer = new ComputeBuffer(areasList.Count, 4);

        _computeShader.SetBuffer(algorythmKernel, "DataBuffer", dataBuffer);
        _computeShader.SetBuffer(algorythmKernel, "RegistryBuffer", registryBuffer);
        _computeShader.SetBuffer(algorythmKernel, "AreasBuffer", areasBuffer);
        _computeShader.SetBuffer(algorythmKernel, "ResultBuffer", resultBuffer);
        _computeShader.SetInt("Divider", 1000);
        _computeShader.SetInt("AreasCount", _allAreas.Count());
        _computeShader.SetInt("SpritesCount", _sprites.Length);

        Debug.LogError($"areas.Length = {areas.Length}");
        Debug.LogError($"registry.Length = {registry.Length}");
        Debug.LogError($"data.Length = {data.Length}");

        Debug.LogError($"_allAreas.Count() = {_allAreas.Count()}");

        var maxOpsCountAllowed = 295000;

        var counts = new List<int>();

        for (int a = 0; a < areasList.Count; a++)
        {
            var area = areasList[a];

            var countForTheArea = 0;
            for (int i = 0; i < 8; i++)
            {
                var sprite = _sprites[i];
                var lastSpriteX = sprite.Length - area.Dimensions.X;
                var lastSpriteY = sprite[0].Length - area.Dimensions.Y;

                countForTheArea += lastSpriteX * lastSpriteY * area.Dimensions.Square;

                //for (int spriteX = 0; spriteX < lastSpriteX; spriteX++)
                //{
                //    for (int spriteY = 0; spriteY < lastSpriteY; spriteY++)
                //    {

                //        for (int areaX = 0; areaX < area.Dimensions.X; areaX++)
                //        {
                //            for (int areaY = 0; areaY < area.Dimensions.Y; areaY++)
                //            {

                //            }
                //        }
                //    }
                //}
            }

            counts.Add(countForTheArea);
        }

        counts = counts.OrderByDescending(c => c).ToList();
        for (int i = 0; i < 100; i++)
        {
            Debug.Log($"area #{i}: ops count - {counts[i]}");
        }

        return result;

        var stopWatch = new System.Diagnostics.Stopwatch();
        stopWatch.Start();

        var iterationsCount = Mathf.CeilToInt(_allAreas.Count() / (float)groupSizeX);
        _computeShader.Dispatch(algorythmKernel, iterationsCount, 1, 1);

        stopWatch.Stop();
        // Get the elapsed time as a TimeSpan value.
        var ts = stopWatch.Elapsed;

        Debug.Log($"Диспатч прошел. Занял он {ts}");

        var resultData = new int[areas.Length];
        resultBuffer.GetData(resultData);

        Debug.Log($"Забрали результат");

        dataBuffer.Dispose();
        registryBuffer.Dispose();
        areasBuffer.Dispose();
        resultBuffer.Dispose();

        var scoresList = new List<int>();
        var testValues1List = new List<int>();
        var testValues2List = new List<int>();

        for (int j = 0; j < /*areasList.Count*/100; j++)
        {
            //var i = j;
            var (count, testValue1, testValue2) = countScoreForArea(areasList[j]);
            scoresList.Add(count * (int)areasList[j].Score);
            testValues1List.Add(testValue1);
            testValues2List.Add(testValue2);
        }

        var resultIsCorrect = true;
        for (int i = 0; i < testValues1List.Count; i++)
        {
            var s = (int)areasList[i].Score;
            var a = resultData[i];
            var b = testValues1List[i];
            if (a != b)
            {
                resultIsCorrect = false;
                break;
            }
        }

        Debug.Log($"Результат проверен. Он {resultIsCorrect}.");

        //Debug.Log($"Cписок составили. Начинаем цикл убирания пикселей...");
        //while (UnprocessedPixels > 0)
        //{
        //    if (_ct.IsCancellationRequested)
        //        break;
        //}

        return result;
    }

    private (int count, int testValue1, int testValue2) countScoreForArea(MyArea area)
    {
        var result = 0;
        var testValue1 = 0;
        var testValue2 = 0;
        for (int i = 0; i < _sprites.Length; i++)
        {
            var sprite = _sprites[i];
            var lastSpriteX = sprite.Length - area.Dimensions.X;
            var lastSpriteY = sprite[0].Length - area.Dimensions.Y;

            //testValue2 += area.SpriteRect.X + area.SpriteRect.Y + area.SpriteRect.Width + area.SpriteRect.Height;

            for (int spriteX = 0; spriteX < lastSpriteX; spriteX++)
            {
                for (int spriteY = 0; spriteY < lastSpriteY; spriteY++)
                {
                    var maybeThis = true;
                    for (int areaX = 0; areaX < area.Dimensions.X; areaX++)
                    {
                        for (int areaY = 0; areaY < area.Dimensions.Y; areaY++)
                        {
                            //testValue2++;
                            var pixelX = spriteX + areaX;
                            var pixelY = spriteY + areaY;
                            var candidatePixel = sprite[pixelX][pixelY];

                            var areaPixel = _sprites[area.SpriteIndex][area.SpriteRect.X + areaX][area.SpriteRect.Y + areaY];
                            testValue1 += areaPixel.Color;
                            testValue2 += candidatePixel.Color;
                            //if (areaPixel.Color != candidatePixel.Color)
                            //{
                            //    maybeThis = false;
                            //    break;
                            //}
                            //if (areaPixel.R != candidatePixel.R || areaPixel.G != candidatePixel.G || areaPixel.B != candidatePixel.B || areaPixel.A != candidatePixel.A)
                            //{
                            //    maybeThis = false;
                            //    break;
                            //}
                        }

                        if (!maybeThis)
                            break;
                    }

                    if (maybeThis)
                        result++;
                }
            }
        }

        //if (result > 1)
        //    Debug.LogError($"result = {result}");

        return (result, testValue1, testValue2);
    }

    public async Task<Dictionary<MyArea, List<MyAreaCoordinates>>> RunCpu()
    {
        Debug.LogError($"Run 1");
        OverallProgressReport.OperationDescription = "Removing areas from picture";
        OverallProgressReport.OperationsCount = UnprocessedPixels;
        var map = new Dictionary<MyArea, List<MyAreaCoordinates>>();

        var currentAreaIndex = 0;

        List<KeyValuePair<int, MyArea>> orderedAreas = null;
        while (UnprocessedPixels > 0)
        {
            Debug.LogError($"Run 2");
            if (_ct.IsCancellationRequested)
                break;
            Debug.LogError($"Run 3");
            if (currentAreaIndex % _areasFreshmentSpan == 0)
                await setAreasAndScores();
            Debug.LogError($"Run 4");

            //После удаления некоторых пикселей рейтинги областей могут меняться - поэтому надо обновлять и переупорядочивать каждый раз.
            if (orderedAreas != null)
            {
                OperationProgressReport.OperationDescription = "Updating volatile scores";
                OperationProgressReport.OperationsCount = _areasVolatilityRange;
                OperationProgressReport.OperationsDone = 0;
                Parallel.For(0, _areasVolatilityRange, (i, loopState) =>
                {
                    if (_ct.IsCancellationRequested)
                        loopState.Break();

                    var invalidAreas = new List<int>();
                    var area = orderedAreas[i].Value;
                    foreach (var kvp in area.Correlations)
                    {
                        var correlation = kvp.Value;
                        if (_mapOfEmptiness.Contains(correlation.Dimensions, correlation.SpriteIndex, correlation.X, correlation.Y))
                            invalidAreas.Add(kvp.Key);
                        else
                        {
                            var sprite = _sprites[correlation.SpriteIndex];
                            var correlatedArea = MyArea.CreateFromSprite(sprite, correlation.SpriteIndex, correlation.X, correlation.Y, correlation.Dimensions);
                            if (correlatedArea.GetHashCode() != area.GetHashCode())
                                invalidAreas.Add(kvp.Key);
                        }
                    }
                    for (int j = 0; j < invalidAreas.Count; j++)
                    {
                        MyAreaCoordinates val;
                        area.Correlations.TryRemove(invalidAreas[j], out val);
                    }
                    invalidAreas.Clear();
                    OperationProgressReport.OperationsDone++;
                });
            }
            Debug.LogError($"Run 5");
            orderedAreas?.Clear();
            orderedAreas = _allAreas.OrderByDescending(kvp => kvp.Value.Correlations.Count * kvp.Value.Score).ToList();
            var currentArea = orderedAreas[0];

            Debug.Log($"Removing area #{currentAreaIndex}: hash {currentArea.Key}, score {currentArea.Value.Correlations.Count * currentArea.Value.Score}");
            var areasRemoved = await applyBestArea(_sprites, currentArea.Key);
            var pixelsRemoved = currentArea.Value.OpaquePixelsCount * areasRemoved.Count;
            map.Add(currentArea.Value, areasRemoved);
            OverallProgressReport.OperationsDone += pixelsRemoved;
            UnprocessedPixels -= pixelsRemoved;
            currentAreaIndex++;
            Debug.LogError($"Run 6");
        }
        Debug.Log($"Done!");

        return map;
    }

    private async Task setAreasAndScores()
    {
        //Debug.LogError($"setAreasAndScores 1");
        _allAreas = await _areaFetcher.FetchAreas(_sprites, _areaSizings, _areaEnumerator, OperationProgressReport);
        //Debug.LogError($"setAreasAndScores 2");
    }

    private int getBestArea(IOrderedEnumerable<KeyValuePair<int, long>> rating) => rating.First().Key;

    private async Task<List<MyAreaCoordinates>> applyBestArea(MyColor[][][] sprites, int bestAreaIndex)
    {
        var result = new List<MyAreaCoordinates>();
        MyArea bestArea;
        if (!_allAreas.TryRemove(bestAreaIndex, out bestArea))
            throw new ApplicationException($"Area is not found!");

        var correlations = bestArea.Correlations;
        foreach (var kvp in correlations)
        {
            var myAreaCoordinates = kvp.Value;
            if (_mapOfEmptiness.Contains(myAreaCoordinates.Dimensions, myAreaCoordinates.SpriteIndex, myAreaCoordinates.X, myAreaCoordinates.Y))
                continue;

            var candidateForErasing = MyArea.CreateFromSprite(sprites[myAreaCoordinates.SpriteIndex], myAreaCoordinates.SpriteIndex, myAreaCoordinates.X, myAreaCoordinates.Y, myAreaCoordinates.Dimensions);
            if (candidateForErasing.GetHashCode() != bestArea.GetHashCode())
                continue;
            MyArea.EraseAreaFromSprite(sprites[myAreaCoordinates.SpriteIndex], myAreaCoordinates.X, myAreaCoordinates.Y, myAreaCoordinates.Dimensions);
            _mapOfEmptiness.MakeEmpty(myAreaCoordinates.Dimensions, myAreaCoordinates.SpriteIndex, myAreaCoordinates.X, myAreaCoordinates.Y);
            result.Add(myAreaCoordinates);
        }

        return result;
    }
}
