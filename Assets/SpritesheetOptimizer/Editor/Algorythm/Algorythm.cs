using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
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

    private readonly MyVector2Float[] _pivots;
    private readonly MyColor[][][] _sprites;
    private readonly IList<ISizingsConfigurator> _sizingsConfigurators;
    private readonly IList<IScoreCounter> _scoreCounters;
    private readonly Type _areaEnumeratorType;
    private readonly ComputeMode _computeMode;

    private readonly ComputeShader _computeShader;

    private ConcurrentDictionary<string, MyArea> _allAreas;
    private MapOfEmptiness _mapOfEmptiness;
    private IEnumerable<MyVector2> _areaSizings;
    private IAreaEnumerator _areaEnumerator;
    private IAreaFetcher _areaFetcher;

    private CancellationToken _ct;

    public Algorythm(MyColor[][][] sprites, MyVector2Float[] pivots, Type areaEnumeratorType, IList<ISizingsConfigurator> sizingConfigurators, IList<IScoreCounter> scoreCounters, int areasFreshmentSpan, int areasVolatilityRange, ComputeMode computeMode)
    {
        OperationProgressReport = new ProgressReport();
        OverallProgressReport = new ProgressReport();

        _areasFreshmentSpan = areasFreshmentSpan;
        _areasVolatilityRange = areasVolatilityRange;

        _pivots = pivots;
        _sprites = sprites;
        _areaEnumeratorType = areaEnumeratorType;
        _sizingsConfigurators = sizingConfigurators;
        _scoreCounters = scoreCounters;

        _computeMode = computeMode;

        _computeShader = Resources.Load<ComputeShader>("SpritesheetOptimizer/Shaders/TheOneForAllShader");
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
            if (sprite[x][y].A != 0)
                result++;
        });
        return result;
    }

    #endregion Initializing

    public struct areaStruct
    {
        public int MetaAndSpriteIndex; //Для индекса спрайта вполне хватит и трех байт. Даже двух, я думаю, хватило бы. Поэтому 1й байт - для мета-информации
        public int XAndY; //Экономим место, т.к. эти буфферы тянут на сотни мегабайт...
        public int WidthAndHeight;

        public areaStruct(int metaAndSpriteIndex, int xAndY, int widthAndHeight)
        {
            MetaAndSpriteIndex = metaAndSpriteIndex;
            XAndY = xAndY;
            WidthAndHeight = widthAndHeight;
        }
    }

    public struct registryStruct
    {
        public int SpritesDataOffset; //С какой позиции в буффере data начинается данный спрайт
        public int WidthAndHeight;

        public registryStruct(int spritesDataOffset, int widthAndHeight)
        {
            SpritesDataOffset = spritesDataOffset;
            WidthAndHeight = widthAndHeight;
        }
    }

    public struct taskStruct
    {
        public int MetaAndSpriteIndex;
        public int SpriteXAndY;
        public int XAndY;
        public int AreasCounter;

        public taskStruct(int metaAndSpriteIndex, int spriteXAndY, int xAndY, int areasCounter)
        {
            MetaAndSpriteIndex = metaAndSpriteIndex;
            SpriteXAndY = spriteXAndY;
            XAndY = xAndY;
            AreasCounter = areasCounter;
        }
    }

    /*
     * Ок, следующее, что нам нужно сделать - это распределить буфферы по частям. Важно сделать так, чтобы закончив с одной частью,
     * нам она больше не понадобилась. Прежде, чем это делать, нужно сначала отрефакторить все, что есть сейчас, мне кажется. Эта функция
     * на 600 строк выглядит весьма устрашающе. Для этого надо разобраться, что эта функция делает.
     * 
     * Сначала она делает проход CPU-версии алгоритма, и получает ее лучший результат.
     * Дальше идет немножко работы для юзер-интерфейса
     * Потом мы заполняем сразу 3 больших структуры данных - registry, data, allPossibleAreas
     * Затем мы создаем экземпляр шейдера и заполняем константы, среди них одна большая - registry
     * Параллельно мы опционально создаем и инициализируем константами цпу-версию шейдера для дебага
     * Потом создаются еще 3 переменные, которые затем переиспользуются в цикле.
     * Потом идет цикл while (UnprocessedPixels > 0)
     *      Сначала проставляются переменные не меняющиеся на протяжении данного прохода цикла - дата
     *      Потом проставляются переменные не меняющиеся на протяжении следующего цикла
     *      Затем идет следующий цикл - foreach (var kvp in allPossibleAreas)
     */

    public class Correlation
    {
        public readonly MySerializableColor[][] Colors;
        public readonly MyAreaCoordinates[] Coordinates;

        public Correlation(MySerializableColor[][] colors, MyAreaCoordinates[] coordinates)
        {
            Colors = colors;
            Coordinates = coordinates;
        }
    }

    public async Task<(Correlation[] correlations, GraphicFileInfo[] testImages)> Run()
    {
        var resultImages = new List<GraphicFileInfo>();
        var stopwatch = new MyStopwatch();

        stopwatch.Start($"The Hole Initialization");
#if CPU
        var bestCpuCalculatedAreaList = await getBestCpuCalculatedArea();
        var bestScoreItem = bestCpuCalculatedAreaList.OrderByDescending(o => o.score * o.count).First();
        var bestCpuScore = bestScoreItem.score * bestScoreItem.count;
        var bestCpuCalculatedArea = bestCpuCalculatedAreaList.Where(o => o.score * o.count == bestCpuScore).OrderByDescending(o => o.area.SpriteRect.X << 16 | o.area.SpriteRect.Y).First();

        Debug.LogError($"CPU Unique areas found: {_allAreas.Count}");
#endif

        var resultList = new List<Correlation>();

        OverallProgressReport.OperationDescription = "Removing areas from picture";
        OverallProgressReport.OperationsCount = UnprocessedPixels;
        stopwatch.Start($"Debug.Log's");
        Debug.Log($"UnprocessedPixels total = {UnprocessedPixels}");
        stopwatch.Stop($"Debug.Log's");
        //{
        //    var opaquePixelsLeft = 0;
        //    for (int i = 0; i < _sprites.Length; i++)
        //    {
        //        for (int x = 0; x < _sprites[i].Length; x++)
        //        {
        //            for (int y = 0; y < _sprites[i][x].Length; y++)
        //            {
        //                if (_sprites[i][x][y].A != 0)
        //                    opaquePixelsLeft++;
        //            }
        //        }
        //    }
        //    Debug.Log($"...а на самом деле UnprocessedPixels = {opaquePixelsLeft}.");
        //}

        var dataSize = 0;
        for (int i = 0; i < _sprites.Length; i++)
            dataSize += _sprites[i].Length * _sprites[i][0].Length;
        var data = new int[dataSize];
        var registry = new registryStruct[_sprites.Length];
        var allPossibleAreas = new Dictionary<MyVector2, List<areaStruct>>();

        var dataOffset = 0;
        var sb = new StringBuilder();
        for (int i = 0; i < _sprites.Length; i++)
        {
            var sprite = _sprites[i];
            var width = sprite.Length;
            var height = sprite[0].Length;

            registry[i] = new registryStruct(dataOffset, width << 16 | height);
            var printsCount = 0;
            var printing = false;
            if (dataOffset == 11728)
            {
                printing = true;
                printsCount++;
            }
            if (dataOffset == 40184)
            {
                printing = true;
                printsCount++;
            }
            if (printing)
                sb.AppendLine($"Printing sprite #{i}:");
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    data[dataOffset++] = sprite[x][y].R << 24 | sprite[x][y].G << 16 | sprite[x][y].B << 8 | sprite[x][y].A;
                    if (printing)
                        sb.AppendLine($"({x},{y}) = {sprite[x][y].R},{sprite[x][y].G},{sprite[x][y].B},{sprite[x][y].A}");

                    foreach (var size in _areaSizings)
                    {
                        if (x > width - size.X)
                            continue;
                        if (y > height - size.Y)
                            continue;

                        if (!allPossibleAreas.ContainsKey(size))
                            allPossibleAreas.Add(size, new List<areaStruct>());
                        var list = allPossibleAreas[size];
                        list.Add(new areaStruct((1 << 24) | (i & 16777215), x << 16 | y, size.X << 16 | size.Y));
                    }
                }
            }
            if (printing)
            {
                File.WriteAllText($"C:\\ABC\\alg-{i}.txt", sb.ToString());
                sb.Clear();
            }
            //if (printsCount == 2)
            //    return result;
        }


        var algorythmKernel = _computeShader.FindKernel("CSMain");
        var (groupSizeX, groupSizeY, groupSizeZ) = _computeShader.GetKernelThreadGroupSizes(algorythmKernel);
         
        var registryBuffer = new ComputeBuffer(_sprites.Length, 8);
        stopwatch.Start($"SetData");
        registryBuffer.SetData(registry);
        stopwatch.Stop($"SetData");

        var maxOpsAllowed = 295000;
        var maxBufferLength = 131072;

        //Проставляем константы.
        _computeShader.SetInt("MaxOpsAllowed", maxOpsAllowed);
        _computeShader.SetInt("SpritesCount", _sprites.Length);
        _computeShader.SetBuffer(algorythmKernel, "RegistryBuffer", registryBuffer);

#if GPUEmu
        var cpuBugTester = new CpuBugTest();
        cpuBugTester.MaxOpsAllowed = maxOpsAllowed;
        cpuBugTester.SpritesCount = _sprites.Length;
        cpuBugTester.RegistryBuffer = registry;
#endif

        var bestOfEachArea = new Dictionary<MyVector2, (int spriteIndex, MyVector2 position, int count, int score, string test)>();
        var chunkCountArrayList = new List<int[]>();

        //var dataBuffer = new ComputeBuffer(dataSize, 4);

        var fullpasses = 0;
        stopwatch.Stop($"The Hole Initialization");
        stopwatch.Start($"The Hole Main Loop");
        while (UnprocessedPixels > 0)
        {
            stopwatch.Start($"The Hole Non GPU Dispatch in the loop");
            //1. Считаем оценки каждой области...

            stopwatch.Start($"SetData");
            var dataBuffer = new ComputeBuffer(dataSize, 4);
            dataBuffer.SetData(data);
            stopwatch.Stop($"SetData");

            //Проставляем переменные, не меняющиеся для данного прохода цикла.
            _computeShader.SetBuffer(algorythmKernel, "DataBuffer", dataBuffer);

#if GPUEmu
            cpuBugTester.DataBuffer = data;
#endif

            var areasCounter = 0;
            var gpuUniqueAreas = new List<areaStruct>();
            stopwatch.Stop($"The Hole Non GPU Dispatch in the loop");
            foreach (var kvp in allPossibleAreas)
            {
                stopwatch.Start($"The Hole Non GPU Dispatch in the loop");
                //Debug.Log($"area #{++areasCounter}");
                //Thread.Sleep(250);
                var size = kvp.Key;
                var areasOfThatSize = kvp.Value.ToArray();

                var zeroes = new int[areasOfThatSize.Length];
                for (int i = 0; i < zeroes.Length; i++)
                    zeroes[i] = 0;

                var areasBuffer = new ComputeBuffer(areasOfThatSize.Length, 12);
                stopwatch.Start($"SetData");
                areasBuffer.SetData(areasOfThatSize);
                stopwatch.Stop($"SetData");

                var tasks = new taskStruct[areasOfThatSize.Length];
                {
                    var initialMask = 1;
                    var metaAndSpriteIndex = initialMask << 24;
                    for (int i = 0; i < tasks.Length; i++)
                        tasks[i] = new taskStruct(metaAndSpriteIndex, 0, 0, 0);
                }

                var tasksBuffer = new ComputeBuffer(areasOfThatSize.Length, 16);
                stopwatch.Start($"SetData");
                tasksBuffer.SetData(tasks);
                stopwatch.Stop($"SetData");

                var countsBuffer = new ComputeBuffer(areasOfThatSize.Length, 4);
                var scoresBuffer = new ComputeBuffer(areasOfThatSize.Length, 4);

                //Проставляем переменные, не меняющиеся для данного размера области.
                _computeShader.SetBuffer(algorythmKernel, "AreasBuffer", areasBuffer);
                _computeShader.SetBuffer(algorythmKernel, "TasksBuffer", tasksBuffer);
                _computeShader.SetBuffer(algorythmKernel, "CountsBuffer", countsBuffer);
                _computeShader.SetBuffer(algorythmKernel, "ScoresBuffer", scoresBuffer);

#if GPUEmu
                cpuBugTester.AreasBuffer = areasOfThatSize;
                cpuBugTester.TasksBuffer = tasks;
                cpuBugTester.CountsBuffer = blockCopy(zeroes);
                cpuBugTester.ScoresBuffer = blockCopy(zeroes);
#else
                var zeroezInstance1 = blockCopy(zeroes);
                stopwatch.Start($"SetData");
                countsBuffer.SetData(zeroezInstance1);
                stopwatch.Stop($"SetData");
                var zeroezInstance2 = blockCopy(zeroes);
                stopwatch.Start($"SetData");
                scoresBuffer.SetData(zeroezInstance2);
                stopwatch.Stop($"SetData");
#endif


                //Проходимся данным размером области по всем возможным пикселам...
                var scores = new int[0];
                var tasksUpdated = new taskStruct[areasOfThatSize.Length];
                var iterationsCount = Mathf.CeilToInt(areasOfThatSize.Length / (float)groupSizeX);
                var passes = 0;
                stopwatch.Stop($"The Hole Non GPU Dispatch in the loop");

                while (true)
                {
                    passes++;

#if GPUEmu
                    for (int i = 0; i < areasOfThatSize.Length; i++)
                        cpuBugTester.Dispatch(i);

                    var allAreasDone = true;
                    for (int i = 0; i < cpuBugTester.TasksBuffer.Length; i++)
                    {
                        if ((cpuBugTester.TasksBuffer[i].MetaAndSpriteIndex >> 24 & 255) == 1)
                        {
                            allAreasDone = false; //Продолжаем пока хотя бы одна область нуждается в обработке
                            break;
                        }
                    }

                    chunkCountArrayList.Add(cpuBugTester.CountsBuffer);
                    cpuBugTester.CountsBuffer = blockCopy(zeroes);

                    if (allAreasDone)
                    {
                        scores = cpuBugTester.ScoresBuffer;
                        cpuBugTester.ScoresBuffer = blockCopy(zeroes);
                        break;
                    }
#else
                    stopwatch.Start($"Dispatch");
                    _computeShader.Dispatch(algorythmKernel, iterationsCount, 1, 1);
                    stopwatch.Stop($"Dispatch");

                    stopwatch.Start($"The Hole Non GPU Dispatch in the loop");

                    stopwatch.Start($"GetData");
                    tasksBuffer.GetData(tasksUpdated);
                    stopwatch.Stop($"GetData");
                    var allAreasDone = true;
                    for (int i = 0; i < tasksUpdated.Length; i++)
                    {
                        if ((tasksUpdated[i].MetaAndSpriteIndex >> 24 & 255) == 1)
                        {
                            allAreasDone = false; //Продолжаем пока хотя бы одна область нуждается в обработке
                            break;
                        }
                    } 

                    var chunkCountsArray = new int[areasOfThatSize.Length];
                    stopwatch.Start($"GetData");
                    countsBuffer.GetData(chunkCountsArray);
                    stopwatch.Stop($"GetData");
                    for (int i = 0; i < chunkCountsArray.Length; i++)
                    {
                        if (chunkCountsArray[i] > 10_000_000)
                        {

                        }
                    }

                    chunkCountArrayList.Add(chunkCountsArray);
                    var zeroezInstance3 = blockCopy(zeroes);
                    stopwatch.Start($"SetData");
                    countsBuffer.SetData(zeroezInstance3);
                    stopwatch.Stop($"SetData");

                    if (allAreasDone)
                    {
                        scores = new int[areasOfThatSize.Length];
                        stopwatch.Start($"GetData");
                        scoresBuffer.GetData(scores); //Напоследок забираем оценки каждой отдельной области
                        stopwatch.Stop($"GetData");
                        var zeroezInstance4 = blockCopy(zeroes);
                        stopwatch.Start($"SetData");
                        scoresBuffer.SetData(zeroezInstance4);
                        stopwatch.Stop($"SetData");
                        break;
                    }
                    stopwatch.Stop($"The Hole Non GPU Dispatch in the loop");
#endif
                }

                stopwatch.Start($"The Hole Non GPU Dispatch in the loop");
                //Debug.Log($"Диспатч прошел. Занял он {ts}. passes = {passes}");

                var totalCounts = new int[areasOfThatSize.Length];
                //var everyCountIs0 = true;
                for (int i = 0; i < totalCounts.Length; i++)
                {
                    var totalScore = 0;
                    for (int j = 0; j < chunkCountArrayList.Count; j++)
                        totalScore += chunkCountArrayList[j][i];
                    //if (totalScore != 0)
                    //    everyCountIs0 = false;
                    totalCounts[i] = totalScore;
                }
                //if (everyCountIs0)
                //{
                //    Debug.LogError($"({size.X},{size.Y}) everyCountIs0");
                //}

                chunkCountArrayList.Clear();

                ////Считаем scores на цпу, потому что gpu почему-то часто тупит с этим
                //areasBuffer.GetData(areasOfThatSize);
                //var everyScoreIs0 = true;
                //var everyCountAndScoreIs0 = true;
                //for (int i = 0; i < areasOfThatSize.Length; i++)
                //{
                //    if ((areasOfThatSize[i].MetaAndSpriteIndex >> 24 & 255) == 0)
                //    {
                //        scores[i] = 0; 
                //        continue;
                //    }
                //    var index = areasOfThatSize[i].MetaAndSpriteIndex & 16777215;
                //    var x = areasOfThatSize[i].XAndY >> 16 & 65535;
                //    var y = areasOfThatSize[i].XAndY & 65535;
                //    var width = areasOfThatSize[i].WidthAndHeight >> 16 & 65535;
                //    var height = areasOfThatSize[i].WidthAndHeight & 65535;

                //    var offset = registry[index].SpritesDataOffset;
                //    var spriteHeight = registry[index].WidthAndHeight & 65535;
                //    var opaquePixels = 0;
                //    for (int xx = 0; xx < width; xx++)
                //    {
                //        for (int yy = 0; yy < height; yy++)
                //        {
                //            var color = data[offset + (x + xx) * spriteHeight + y + yy];
                //            if ((color & 255) != 0)
                //                opaquePixels++;
                //        }
                //    }

                //    var square = width * height;
                //    //scores[i] = (int)(opaquePixels * opaquePixels * opaquePixels / square);
                //    var score = Mathf.FloorToInt(Mathf.Pow(opaquePixels, 3f) / square);
                //    if (score != 0)
                //        everyScoreIs0 = false;
                //    if (score != 0 && totalCounts[i] != 0)
                //        everyCountAndScoreIs0 = false;
                //    scores[i] = score;
                //}

                //if (everyCountAndScoreIs0)
                //{
                //    Debug.LogError($"({size.X},{size.Y}) everyCountAndScoreIs0");
                //}

                //if (everyScoreIs0)
                //{ 
                //    Debug.LogError($"({size.X},{size.Y}) everyScoreIs0");
                //    if (size.X == 1 && size.Y == 1)
                //    {
                //        for (int i = 0; i < areasOfThatSize.Length; i++)
                //        {
                //            if ((areasOfThatSize[i].MetaAndSpriteIndex >> 24 & 255) == 0)
                //            {
                //                scores[i] = 0;
                //                continue;
                //            }
                //            var index = areasOfThatSize[i].MetaAndSpriteIndex & 16777215;
                //            var x = areasOfThatSize[i].XAndY >> 16 & 65535;
                //            var y = areasOfThatSize[i].XAndY & 65535;
                //            var width = areasOfThatSize[i].WidthAndHeight >> 16 & 65535;
                //            var height = areasOfThatSize[i].WidthAndHeight & 65535;

                //            var offset = registry[index].SpritesDataOffset;
                //            var spriteHeight = registry[index].WidthAndHeight & 65535;
                //            var opaquePixels = 0;
                //            for (int xx = 0; xx < width; xx++)
                //            {
                //                for (int yy = 0; yy < height; yy++)
                //                {
                //                    var color = data[offset + (x + xx) * spriteHeight + y + yy];
                //                    if ((color & 255) != 0)
                //                        opaquePixels++;
                //                }
                //            }

                //            var square = width * height;
                //            //scores[i] = (int)(opaquePixels * opaquePixels * opaquePixels / square);
                //            var score = Mathf.FloorToInt(Mathf.Pow(opaquePixels, 3f) / square);
                //            if (score != 0)
                //                everyScoreIs0 = false;
                //            scores[i] = score;
                //        }
                //    }
                //}

                var maxTotalScore = int.MinValue;
                var maxTotalScoreIndex = -1;
                var maxTotalScorePos = 0;

                //for (int i = 0; i < scores.Length; i++)
                //{
                //    if (scores[i] > 0)
                //        gpuUniqueAreas.Add(areasOfThatSize[i]);
                //}

                for (int i = 0; i < totalCounts.Length; i++)
                {
                    var totalScore = totalCounts[i] * scores[i];
                    if (totalScore > maxTotalScore)
                    {
                        maxTotalScore = totalScore;
                        maxTotalScoreIndex = i;
                        maxTotalScorePos = areasOfThatSize[i].XAndY;
                    }
                    else if (totalScore == maxTotalScore)
                    {
                        if (areasOfThatSize[i].XAndY > maxTotalScorePos)
                        {
                            maxTotalScore = totalScore;
                            maxTotalScoreIndex = i;
                            maxTotalScorePos = areasOfThatSize[i].XAndY;
                        }
                    }
                }

                bestOfEachArea.Add(size, (
                    spriteIndex: areasOfThatSize[maxTotalScoreIndex].MetaAndSpriteIndex & 16777215,
                    position: new MyVector2(areasOfThatSize[maxTotalScoreIndex].XAndY >> 16 & 65535, areasOfThatSize[maxTotalScoreIndex].XAndY & 65535),
                    count: totalCounts[maxTotalScoreIndex],
                    score: scores[maxTotalScoreIndex],
                    test: tasksUpdated[maxTotalScoreIndex].AreasCounter.ToString()
                    )
                );

                //Восстанавливаем флаги на областях
                for (int i = 0; i < areasOfThatSize.Length; i++)
                    areasOfThatSize[i].MetaAndSpriteIndex = 1 << 24 | (areasOfThatSize[i].MetaAndSpriteIndex & 16777215);

                areasBuffer.Dispose();
                tasksBuffer.Dispose();
                countsBuffer.Dispose();
                scoresBuffer.Dispose();
                stopwatch.Stop($"The Hole Non GPU Dispatch in the loop");
            }
            stopwatch.Start($"The Hole Non GPU Dispatch in the loop");

            //Debug.LogError($"GPU Unique areas found: {gpuUniqueAreas.Count}");

#if CPU
            //Проверяем совпадение кол-ва областей
            var areasCounts = new Dictionary<MyVector2, int>();
            var areasOk = true;
            for (int i = 0; i < bestCpuCalculatedAreaList.Count; i++)
            {
                var currentArea = bestCpuCalculatedAreaList[i];
                var size = new MyVector2(currentArea.area.SpriteRect.Width, currentArea.area.SpriteRect.Height);
                var areasCount = currentArea.test;
                if (!areasCounts.ContainsKey(size))
                    areasCounts.Add(size, areasCount);
                if (areasCounts[size] != areasCount)
                {
                    areasOk = false;
                    break;
                }
            }
            if (areasOk)
                Debug.LogError($"Areas counts of cpu results are consistent!");
            else
                Debug.LogError($"Areas counts of cpu results are inconsistent!");


            foreach (var kvp in bestOfEachArea)
            {
                var gpuAreaCounts = kvp.Value.test;
                var cpuAreaCounts = areasCounts[kvp.Key].ToString();
                if (gpuAreaCounts != cpuAreaCounts)
                    Debug.LogError($"Area counts of gpu ({gpuAreaCounts}) and cpu ({cpuAreaCounts}) are incosistent!");
            }
            Debug.LogError($"Area counts checking done!");


            //Проверяем, что gpu области присутствуют в списке цпу областей
            var allAreasExists = true;
            foreach (var kvp in bestOfEachArea)
            {
                var areaFound = false;
                for (int i = 0; i < bestCpuCalculatedAreaList.Count; i++)
                {
                    var currentArea = bestCpuCalculatedAreaList[i].area;
                    if (currentArea.Dimensions.X == kvp.Key.X &&
                        currentArea.Dimensions.Y == kvp.Key.Y)
                    { 
                        var areaTheSame = true;
                        for (int x = 0; x < currentArea.Dimensions.X; x++)
                        {
                            for (int y = 0; y < currentArea.Dimensions.Y; y++)
                            {
                                var pixelA = _sprites[currentArea.SpriteIndex][currentArea.SpriteRect.X + x][currentArea.SpriteRect.Y + y];
                                var pixelB = _sprites[kvp.Value.spriteIndex][kvp.Value.position.X + x][kvp.Value.position.Y + y];
                                if (pixelA.GetHashCode() != pixelB.GetHashCode())
                                {
                                    areaTheSame = false;
                                    break;
                                }
                            }
                        }

                        if (areaTheSame)
                            areaFound = true;
                        break;
                    }
                }
                if (!areaFound)
                {
                    allAreasExists = false;
                    break;
                }
            }

            Debug.LogError($"allAreasExists = {allAreasExists}");


            //Проверяем действительно ли gpu области все уникальны.

            var gpuAreaHashes = new Dictionary<string, areaStruct>();
            sb = new StringBuilder();
            for (int i = 0; i < gpuUniqueAreas.Count; i++)
            {
                var area = gpuUniqueAreas[i];
                var x = area.XAndY >> 16 & 65535;
                var y = area.XAndY & 65535;
                var width = area.WidthAndHeight >> 16 & 65535;
                var height = area.WidthAndHeight & 65535;
                var index = area.MetaAndSpriteIndex & 16777215;

                //sb.Append(width);
                //sb.Append(height);
                for (int xx = 0; xx < width; xx++)
                {
                    sb.Append('_');
                    sb.Append(xx);
                    for (int yy = 0; yy < height; yy++)
                    {
                        var pixel = _sprites[index][x + xx][y + yy];

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
                var key = sb.ToString();
                if (gpuAreaHashes.ContainsKey(key))
                {
                    var repeatArea = gpuAreaHashes[key];
                    Debug.LogError($"repeating area:");
                    Debug.LogError($"index: {repeatArea.MetaAndSpriteIndex & 16777215}");
                    Debug.LogError($"x: {repeatArea.XAndY >> 16 & 65535}");
                    Debug.LogError($"y: {repeatArea.XAndY & 65535}");
                    Debug.LogError($"width: {repeatArea.WidthAndHeight >> 16 & 65535}");
                    Debug.LogError($"height: {repeatArea.WidthAndHeight & 65535}");

                    Debug.LogError($"");
                    Debug.LogError($"new area:");
                    Debug.LogError($"index: {area.MetaAndSpriteIndex & 16777215}");
                    Debug.LogError($"x: {area.XAndY >> 16 & 65535}");
                    Debug.LogError($"y: {area.XAndY & 65535}");
                    Debug.LogError($"width: {area.WidthAndHeight >> 16 & 65535}");
                    Debug.LogError($"height: {area.WidthAndHeight & 65535}");
                }
                gpuAreaHashes.Add(key, area);
                sb.Clear();
            }

            Debug.LogError($"Everything's ok, there's no exceptions, gpu areas are unique");
            Debug.LogError($"Starting cpu areas checking");

            var cpuAreaHashes = new Dictionary<string, MyArea>();
            foreach (var kvp in _allAreas)
            {
                var area = kvp.Value;
                var x = area.SpriteRect.X;
                var y = area.SpriteRect.Y;
                var width = area.SpriteRect.Width;
                var height = area.SpriteRect.Height;
                var index = area.SpriteIndex;
                for (int xx = 0; xx < width; xx++)
                {
                    sb.Append('_');
                    sb.Append(xx);
                    for (int yy = 0; yy < height; yy++)
                    {
                        var pixel = _sprites[index][x + xx][y + yy];

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
                var key = sb.ToString();
                if (cpuAreaHashes.ContainsKey(key))
                {
                    var repeatArea = cpuAreaHashes[key];
                    Debug.LogError($"repeating area:");
                    Debug.LogError($"index: {repeatArea.SpriteIndex}");
                    Debug.LogError($"x: {repeatArea.SpriteRect.X}");
                    Debug.LogError($"y: {repeatArea.SpriteRect.Y}");
                    Debug.LogError($"width: {repeatArea.SpriteRect.Width}");
                    Debug.LogError($"height: {repeatArea.SpriteRect.Height}");

                    Debug.LogError($"");
                    Debug.LogError($"new area:");
                    Debug.LogError($"index: {area.SpriteIndex}");
                    Debug.LogError($"x: {area.SpriteRect.X}");
                    Debug.LogError($"y: {area.SpriteRect.Y}");
                    Debug.LogError($"width: {area.SpriteRect.Width}");
                    Debug.LogError($"height: {area.SpriteRect.Height}");
                }
                cpuAreaHashes.Add(key, area);
                sb.Clear();
            }

            Debug.LogError($"Everything's ok, there's no exceptions, cpu areas are unique");
#endif

            var bestScoreKvp = bestOfEachArea.OrderByDescending(kvp => kvp.Value.count * kvp.Value.score).First();
            var bestScore = bestScoreKvp.Value.score * bestScoreKvp.Value.count;

            var finalOfTheBest = bestOfEachArea.Where(kvp => kvp.Value.count * kvp.Value.score == bestScore).OrderByDescending(kvp => kvp.Value.position.X << 16 | kvp.Value.position.Y).First();

            // 1а. Проверяем не надурили ли мы с алгоритмом

            //bestOfTheBest = 32, (3,15), (6,5): s (900) * c (535) = 481500 (areas 39312). bestCpuCalculatedArea = 33, (0,13), (6,5): s (900) * c (535) = 481500 (areas 39312)

            //bestOfTheBest = 14, (0,68), (5,4): s (625) * c (14) = 8750 (areas 108694). bestCpuCalculatedArea = 5, (32,15), (8,8): s (4096) * c (1) = 4096 (areas 97143)

            //bestOfTheBest = 0, (19,81), (2,3): s (36) * c (3855) = 138780 (areas 72384). bestCpuCalculatedArea = 0, (15,24), (8,7): s (3136) * c (2371) = 7435456 (areas 61088)


#if CPU
            Debug.Log($"bestOfTheBest = {finalOfTheBest.Value.spriteIndex}, ({finalOfTheBest.Value.position.X},{finalOfTheBest.Value.position.Y}), ({finalOfTheBest.Key.X},{finalOfTheBest.Key.Y}): s ({finalOfTheBest.Value.score}) * c ({finalOfTheBest.Value.count}) = {finalOfTheBest.Value.score * finalOfTheBest.Value.count} (areas {finalOfTheBest.Value.test}). bestCpuCalculatedArea = {bestCpuCalculatedArea.area.SpriteIndex}, ({bestCpuCalculatedArea.area.SpriteRect.X},{bestCpuCalculatedArea.area.SpriteRect.Y}), ({bestCpuCalculatedArea.area.SpriteRect.Width},{bestCpuCalculatedArea.area.SpriteRect.Height}): s ({bestCpuCalculatedArea.score}) * c ({bestCpuCalculatedArea.count}) = {bestCpuCalculatedArea.score * bestCpuCalculatedArea.count} (areas {bestCpuCalculatedArea.test})");
#else
            stopwatch.Start($"Debug.Log's");
            Debug.Log($"bestOfTheBest = {finalOfTheBest.Value.spriteIndex}, ({finalOfTheBest.Value.position.X},{finalOfTheBest.Value.position.Y}), ({finalOfTheBest.Key.X},{finalOfTheBest.Key.Y}): s ({finalOfTheBest.Value.score}) * c ({finalOfTheBest.Value.count}) = {finalOfTheBest.Value.score * finalOfTheBest.Value.count} (areas {finalOfTheBest.Value.test}).");
            stopwatch.Stop($"Debug.Log's");

            //if (finalOfTheBest.Value.score == 0 && finalOfTheBest.Value.count == 0)
            //{ 
            //    Debug.LogError($"Все по нулям почему-то. Проверяем, правильно ли это.");

            //    var reg1 = registry[finalOfTheBest.Value.spriteIndex];
            //    var h = reg1.WidthAndHeight & 65535;
            //    var op = 0;
            //    for (int x = 0; x < finalOfTheBest.Key.X; x++)
            //    {
            //        for (int y = 0; y < finalOfTheBest.Key.Y; y++)
            //        {
            //            var pixel = data[reg1.SpritesDataOffset + x * h + y];
            //            if ((pixel & 255) != 0)
            //                op++;
            //        }
            //    }

            //    Debug.LogError($"А на самом деле эта область совсем не пустая, у нее - {op} непрозрачных пикселей!");

            //    Debug.LogError($"И, если посчитать, сколько у нее совпадений, то получится, что их {getCoincidentsCount(finalOfTheBest, registry, data)}");

            //    var newPixels = new MyColor[_sprites.Length][][];
            //    for (int i = 0; i < _sprites.Length; i++)
            //    {
            //        newPixels[i] = new MyColor[_sprites[i].Length][];
            //        for (int x = 0; x < newPixels[i].Length; x++)
            //        {
            //            newPixels[i][x] = new MyColor[_sprites[i][x].Length];
            //            for (int y = 0; y < newPixels[i][x].Length; y++)
            //            {
            //                newPixels[i][x][y] = _sprites[i][x][y];
            //            }
            //        }
            //    }

            //    for (int i = 0; i < _sprites.Length; i++)
            //    {
            //        var reg = registry[i];
            //        var width = reg.WidthAndHeight << 16 & 65535;
            //        var height = reg.WidthAndHeight & 65535;

            //        for (int x = 0; x < width; x++)
            //        {
            //            for (int y = 0; y < height; y++)
            //            {
            //                var color = data[reg.SpritesDataOffset + x * height + y];
            //                _sprites[i][x][y] = new MyColor(color);
            //            }
            //        }
            //    }
            //    Debug.LogError($"!!! Начинаем сначала.");
            //    var result = await Run();

            //    return result;
            //    var opaquePixelsLeft = 0;
            //    for (int i = 0; i < data.Length; i++)
            //    {
            //        if ((data[i] & 255) != 0) 
            //        {
            //            opaquePixelsLeft++;
            //        }
            //    }
            //    if (opaquePixelsLeft > 0)
            //        throw new ApplicationException($"Неправильно - осталось {opaquePixelsLeft} непрозрачных пикселей, а значит не может быть чтобы лучшая область имела score и count равные 0!");
            //}

            bestOfEachArea.Clear();

#endif

#if CPU
            Debug.Log($"Begin comparing best areas");
            var valid = true;
            for (int x = 0; x < finalOfTheBest.Key.X; x++)
            {
                for (int y = 0; y < finalOfTheBest.Key.Y; y++)
                {
                    var aXx = finalOfTheBest.Value.position.X + x;
                    var aYy = finalOfTheBest.Value.position.Y + y;
                    var a = _sprites[finalOfTheBest.Value.spriteIndex][aXx][aYy];
                    var bXx = bestCpuCalculatedArea.area.SpriteRect.X + x;
                    var bYy = bestCpuCalculatedArea.area.SpriteRect.Y + y;
                    var b = _sprites[bestCpuCalculatedArea.area.SpriteIndex][bXx][bYy];
                    if (a.GetHashCode() != b.GetHashCode())
                    {
                        valid = false;
                        break;
                    }
                }

                if (!valid)
                    break;
            }

            Debug.Log($"Результат: {valid}");
#endif

            //2. у нас есть победитель - забираем его данные вхождения из данных! 

            //var theWinnerArea = MyArea.CreateFromSprite(_sprites[finalOfTheBest.Value.spriteIndex], finalOfTheBest.Value.spriteIndex, finalOfTheBest.Value.position.X, finalOfTheBest.Value.position.Y, finalOfTheBest.Key);
            stopwatch.Start($"Areas removing from data");
            var opaqueCount = 0;
            var theWinnerAreaSpriteDataOffset = registry[finalOfTheBest.Value.spriteIndex].SpritesDataOffset;
            var theWinnerAreaSpriteHeight = registry[finalOfTheBest.Value.spriteIndex].WidthAndHeight & 65535;
            for (int areaX = 0; areaX < finalOfTheBest.Key.X; areaX++)
            {
                for (int areaY = 0; areaY < finalOfTheBest.Key.Y; areaY++)
                {
                    var pixelX = finalOfTheBest.Value.position.X + areaX;
                    var pixelY = finalOfTheBest.Value.position.Y + areaY;
                    var pixel = data[theWinnerAreaSpriteDataOffset + pixelX * theWinnerAreaSpriteHeight + pixelY];
                    if ((pixel & 255) != 0)
                        opaqueCount++;
                }
            }
            stopwatch.Start($"Debug.Log's");
            Debug.Log($"theWinnerArea opaque count: {opaqueCount}");
            stopwatch.Stop($"Debug.Log's");

            var areaColors = new MySerializableColor[finalOfTheBest.Key.X][];
            for (int x = 0; x < areaColors.Length; x++)
            {
                areaColors[x] = new MySerializableColor[finalOfTheBest.Key.Y];
                for (int y = 0; y < areaColors[x].Length; y++)
                {
                    var color = data[theWinnerAreaSpriteDataOffset + (finalOfTheBest.Value.position.X + x) * theWinnerAreaSpriteHeight + (finalOfTheBest.Value.position.Y + y)];
                    areaColors[x][y] = new MySerializableColor(color);
                    Debug.Log($"data = {color}. Color = {areaColors[x][y].R},{areaColors[x][y].G},{areaColors[x][y].B},{areaColors[x][y].A}");
                }
            }

            var listOfCorrelations = new ConcurrentBag<MyAreaCoordinates>();
            var pixelsRemoved = 0;
            var actualPixelsRemoved = 0;
            var @lock = new object();
            //var @lock2 = new object();
            //for (int spriteIndex = 0; spriteIndex < _sprites.Length; spriteIndex++)

            resultImages.AddRange(saveCurrentVersionOfData(fullpasses, data, registry));

            Parallel.For(0, _sprites.Length, (spriteIndex, loopState) =>
            {
                var sprite = _sprites[spriteIndex];
                var spriteWidth = sprite.Length;
                var spriteHeight = sprite[0].Length;
                var lastSpriteX = spriteWidth - finalOfTheBest.Key.X + 1;
                var lastSpriteY = spriteHeight - finalOfTheBest.Key.Y + 1;

                var spriteDataOffset = registry[spriteIndex].SpritesDataOffset;

                for (int spriteX = 0; spriteX < lastSpriteX; spriteX++)
                {
                    for (int spriteY = 0; spriteY < lastSpriteY; spriteY++)
                    {
                        var maybeThis = true;
                        //var atLeastOneOpaque = false;
                        for (int areaX = 0; areaX < finalOfTheBest.Key.X; areaX++)
                        {
                            for (int areaY = 0; areaY < finalOfTheBest.Key.Y; areaY++)
                            {
                                var pixelX = spriteX + areaX;
                                var pixelY = spriteY + areaY;

                                //var candidatePixel = new MyColor(data[spriteDataOffset + pixelX * spriteHeight + pixelY]);
                                var candidatePixel = data[spriteDataOffset + pixelX * spriteHeight + pixelY];
                                var areaPixel = areaColors[areaX][areaY];

                                if ((candidatePixel & 255) == 0 && areaPixel.A == 0) //Если пиксель прозрачный - нам не важны все остальные цветовые каналы этого пикселя.
                                    continue;

                                //atLeastOneOpaque = true;
                                if (areaPixel.Color != candidatePixel)
                                {
                                    maybeThis = false;
                                    break;
                                }
                            }

                            if (!maybeThis)
                                break;
                        }

                        if (maybeThis/* && atLeastOneOpaque*/)
                        {
                            var newCorrelation = new MyAreaCoordinates(spriteIndex, spriteX, spriteY, finalOfTheBest.Key.X, finalOfTheBest.Key.Y);
                            //MyArea.EraseAreaFromSprite(sprite, spriteX, spriteY, theWinnerArea.Dimensions);
                            lock (@lock)
                                pixelsRemoved += opaqueCount;
                            listOfCorrelations.Add(newCorrelation);

                            for (int areaX = 0; areaX < finalOfTheBest.Key.X; areaX++)
                            {
                                for (int areaY = 0; areaY < finalOfTheBest.Key.Y; areaY++)
                                {
                                    var pixelX = spriteX + areaX;
                                    var pixelY = spriteY + areaY;
                                    var pixel = data[spriteDataOffset + pixelX * spriteHeight + pixelY];
                                    //if ((pixel & 255) != 0)
                                    //    lock (@lock)
                                    //        actualPixelsRemoved++;
                                    //lock(@lock2)
                                        data[spriteDataOffset + pixelX * spriteHeight + pixelY] = 0;
                                } 
                            }
                        }
                    }
                }
            });
            stopwatch.Stop($"Areas removing from data");

            //for (int x = 0; x < areaColors.Length; x++)
            //{
            //    for (int y = 0; y < areaColors[x].Length; y++)
            //    {
            //        Debug.Log($"areaColors[{x}][{y}] = {areaColors[x][y].R},{areaColors[x][y].G},{areaColors[x][y].B},{areaColors[x][y].A}");
            //    }
            //}
            resultList.Add(new Correlation(areaColors, listOfCorrelations.ToArray()));

            UnprocessedPixels -= pixelsRemoved;

            stopwatch.Start($"Debug.Log's");
            Debug.Log($"Удалено: {pixelsRemoved}. actualPixelsRemoved = {actualPixelsRemoved}. Осталось {UnprocessedPixels}");
            stopwatch.Stop($"Debug.Log's");

            //{
            //    var opaquePixelsLeft = 0;
            //    for (int i = 0; i < data.Length; i++)
            //    {
            //        if ((data[i] & 255) != 0)
            //        {
            //            opaquePixelsLeft++;
            //        }
            //    }
            //    Debug.Log($"...а на самом деле осталось {opaquePixelsLeft}.");
            //}

            stopwatch.Start($"Debug.Log's");
            Debug.Log($"Проход {fullpasses + 1} завершен.");
            stopwatch.Stop($"Debug.Log's");
            stopwatch.Stop($"The Hole Non GPU Dispatch in the loop");

            dataBuffer.Dispose();

            if (++fullpasses >= 300)
                break;
        }
        stopwatch.Stop($"The Hole Main Loop");

        registryBuffer.Dispose();
        //dataBuffer.Dispose();

        Debug.Log(stopwatch.PrintResults());

        return (resultList.ToArray(), resultImages.ToArray());
    }

    private GraphicFileInfo[] saveCurrentVersionOfData(int pass, int[] data, registryStruct[] registry)
    {
        var result = new GraphicFileInfo[_sprites.Length];
        var pathBegin = $"Pass_{pass}";
        for (int i = 0; i < _sprites.Length; i++)
        {
            var fileInfo = new GraphicFileInfo();
            var sprite = _sprites[i];
            var spriteWidth = sprite.Length;
            var spriteHeight = sprite[0].Length;

            fileInfo.Width = spriteWidth;
            fileInfo.Height = spriteHeight;

            var spriteDataOffset = registry[i].SpritesDataOffset;

            var colors = new Color[spriteWidth * spriteHeight];
            for (int spriteX = 0; spriteX < spriteWidth; spriteX++)
            {
                for (int spriteY = 0; spriteY < spriteHeight; spriteY++)
                {
                    var pixel = data[spriteDataOffset + spriteX * spriteHeight + spriteY];
                    var color = new Color(((pixel >> 24) & 255) / 255f, ((pixel >> 16) & 255) / 255f, ((pixel >> 8) & 255) / 255f, (pixel & 255) / 255f);
                    colors[spriteY * spriteWidth + spriteX] = color;
                }
            }
            fileInfo.Colors = colors;
            fileInfo.filePath = Path.Combine(pathBegin, $"sprite_{i}.png");
            result[i] = fileInfo;
        }
        return result;
    }

    private int[] blockCopy(int[] source)
    {
        var result = new int[source.Length];
        Buffer.BlockCopy(source, 0, result, 0, source.Length * 4);
        return result;
    }

    private int getCoincidentsCount(KeyValuePair<MyVector2, (int spriteIndex, MyVector2 position, int count, int score, string test)> kvp, registryStruct[] registry, int[] data)
    {
        var areaSpriteDataOffset = registry[kvp.Value.spriteIndex].SpritesDataOffset;
        var areaSpriteHeight = registry[kvp.Value.spriteIndex].WidthAndHeight & 65535;

        var result = 0;
        for (int spriteIndex = 0; spriteIndex < _sprites.Length; spriteIndex++)
        //Parallel.For(0, _sprites.Length, (spriteIndex, loopState) =>
        {
            var reg = registry[spriteIndex];
            var spriteWidth = reg.WidthAndHeight >> 16 & 65535;
            var spriteHeight = reg.WidthAndHeight & 65535;
            var lastSpriteX = spriteWidth - kvp.Key.X + 1;
            var lastSpriteY = spriteHeight - kvp.Key.Y + 1;

            var spriteDataOffset = registry[spriteIndex].SpritesDataOffset;

            for (int spriteX = 0; spriteX < lastSpriteX; spriteX++)
            {
                for (int spriteY = 0; spriteY < lastSpriteY; spriteY++)
                {
                    var maybeThis = true;
                    //var atLeastOneOpaque = false;
                    for (int areaX = 0; areaX < kvp.Key.X; areaX++)
                    {
                        for (int areaY = 0; areaY < kvp.Key.Y; areaY++)
                        {
                            var pixelX = spriteX + areaX;
                            var pixelY = spriteY + areaY;
                            var sourceX = kvp.Value.position.X + areaX;
                            var sourceY = kvp.Value.position.Y + areaY;

                            //var candidatePixel = new MyColor(data[spriteDataOffset + pixelX * spriteHeight + pixelY]);
                            var candidatePixel = data[spriteDataOffset + pixelX * spriteHeight + pixelY];
                            var areaPixel = data[areaSpriteDataOffset + sourceX * areaSpriteHeight + sourceY];

                            if ((candidatePixel & 255) == 0 && (areaPixel & 255) == 0) //Если пиксель прозрачный - нам не важны все остальные цветовые каналы этого пикселя.
                                continue;

                            //atLeastOneOpaque = true;
                            if (areaPixel != candidatePixel)
                            {
                                maybeThis = false;
                                break;
                            }
                        }

                        if (!maybeThis)
                            break;
                    }

                    if (maybeThis)
                        result++;
                }
            }
        }

        return result;
    }

    private string printChunkCounts(List<int[]> chunkCountArrayList, int maxTotalScoreIndex)
    {
        var sb = new StringBuilder();

        for (int i = 0; i < chunkCountArrayList.Count; i++)
        {
            if (i > 0)
                sb.Append($", ");
            sb.Append(chunkCountArrayList[i][maxTotalScoreIndex]);
        }

        return sb.ToString();
    }

    private async Task<List<(MyArea area, int score, int count, int test)>> getBestCpuCalculatedArea()
    {
        Debug.Log($"Составляем список всех уникальных областей...");
        await setAreasAndScores();
        Debug.Log($"Список составили, начали составлять буфер");

        var scoresList = new List<(MyArea area, int score, int count, int test)>();
        var testValues1List = new List<int>();
        var testValues2List = new List<int>();
        var areasList = _allAreas.Select(kvp => kvp.Value).ToList();

        for (int m = 0; m < areasList.Count; m++)
        {
            var (count, testValue1, testValue2) = countScoreForArea(areasList[m]);
            var c = count;
            var ar = areasList[m];
            var s = (int)ar.Score;
            scoresList.Add((ar, s, c, testValue1));
        }

        return scoresList;
    }

    public async Task<List<MyArea>> RunLegacy()
    {
        OverallProgressReport.OperationDescription = "Removing areas from picture";
        OverallProgressReport.OperationsCount = UnprocessedPixels;
        var result = new List<MyArea>();

        Debug.Log($"-3");
        Thread.Sleep(1000);

        Debug.Log($"Составляем список всех уникальных областей...");
        await setAreasAndScores();
        Debug.Log($"Список составили, начали составлять буфер");

        Debug.Log($"-2");
        Thread.Sleep(1000);

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

        Debug.Log($"-1");
        Thread.Sleep(1000);

        var algorythmKernel = _computeShader.FindKernel("CSMain");
        var (groupSizeX, groupSizeY, groupSizeZ) = _computeShader.GetKernelThreadGroupSizes(algorythmKernel);

        var registryBuffer = new ComputeBuffer(_sprites.Length, 8);
        registryBuffer.SetData(registry);

        var areasList = _allAreas.Select(kvp => kvp.Value).ToList();
        var areas = new areaStruct[areasList.Count];
        for (int i = 0; i < areasList.Count; i++)
        {
            var area = areasList[i];
            var initialMask = 1;
            var metaAndSpriteIndex = initialMask << 24 | area.SpriteIndex & 16777215;
            areas[i] = new areaStruct(metaAndSpriteIndex, area.SpriteRect.X << 16 | area.SpriteRect.Y, area.SpriteRect.Width << 16 | area.SpriteRect.Height);
        }

        Debug.Log($"0");
        Thread.Sleep(1000);

        var resultBuffer = new ComputeBuffer(areasList.Count, 4);

        var tasks = new taskStruct[areasList.Count];
        {
            var initialMask = 1;
            var metaAndSpriteIndex = initialMask << 24;
            for (int i = 0; i < tasks.Length; i++)
                tasks[i] = new taskStruct(metaAndSpriteIndex, 0, 0, 0);
        }
        var tasksBuffer = new ComputeBuffer(tasks.Length, 16);
        tasksBuffer.SetData(tasks);

        _computeShader.SetBuffer(algorythmKernel, "RegistryBuffer", registryBuffer); //Это мы кидаем и никогда не меняем
        _computeShader.SetBuffer(algorythmKernel, "ResultBuffer", resultBuffer); //Это мы кидаем и только забираем
        _computeShader.SetBuffer(algorythmKernel, "TasksBuffer", tasksBuffer); //А это мы устанавливаем вначале и дальше он сам по себе, обновляется каждый диспатч
        //_computeShader.SetInt("AreasCount", areasList.Count);
        _computeShader.SetInt("SpritesCount", _sprites.Length);
        _computeShader.SetInt("MaxOpsAllowed", 295000);
        //_computeShader.SetInt("Divider", 1000);

        var dataBuffer = new ComputeBuffer(dataSize, 4);
        var areasBuffer = new ComputeBuffer(areasList.Count, 12);

        //var maxOpsCountAllowed = 150000;
        //var maxOpsCountAllowed = 295000;

        while (UnprocessedPixels > 0)
        {
            if (_ct.IsCancellationRequested)
                break;

            // 1. Устанавливаем самые свежие версии данных и областей

            dataBuffer.SetData(data);
            areasBuffer.SetData(areas);

            _computeShader.SetBuffer(algorythmKernel, "DataBuffer", dataBuffer);
            _computeShader.SetBuffer(algorythmKernel, "AreasBuffer", areasBuffer);
            Debug.Log($"1");
            Thread.Sleep(1000);

            // 2. Разделяем задачу на чанки, чтобы видюха смогла все это переварить

            //var counts = new List<int>();

            //for (int a = 0; a < areasList.Count; a++)
            //{
            //    var area = areasList[a];

            //    var countForTheArea = 0;
            //    for (int i = 0; i < _sprites.Length; i++)
            //    {
            //        var sprite = _sprites[i];
            //        var lastSpriteX = sprite.Length - area.Dimensions.X;
            //        var lastSpriteY = sprite[0].Length - area.Dimensions.Y;

            //        countForTheArea += lastSpriteX * lastSpriteY * area.Dimensions.Square;
            //    }

            //    counts.Add(countForTheArea);
            //}

            //var maxCount = counts.OrderByDescending(c => c).First();
            //var chunksNumber = Mathf.CeilToInt(maxCount / (float)maxOpsCountAllowed);
            //var span = Mathf.CeilToInt(_sprites.Length / chunksNumber);
            //Debug.LogError($"maxCount = {maxCount}, span = {span}, chunksNumber = {chunksNumber}, sprites.Length = {_sprites.Length}");

            //Debug.Log($"2");
            //Thread.Sleep(1000);


            // 3. Диспатчим и забираем результат

            var stopWatch = new System.Diagnostics.Stopwatch();
            stopWatch.Start();

            var chunkResultsList = new List<int[]>();
            var iterationsCount = Mathf.CeilToInt(areasList.Count / (float)groupSizeX);
            var passes = 0;
            while (true)
            {
                passes++;
                _computeShader.Dispatch(algorythmKernel, iterationsCount, 1, 1);
                var tasksUpdated = new taskStruct[areas.Length];
                tasksBuffer.GetData(tasksUpdated);
                var allAreasDone = true;
                for (int i = 0; i < tasksUpdated.Length; i++)
                {
                    if ((tasksUpdated[i].MetaAndSpriteIndex >> 24 & 255) == 1)
                    {
                        allAreasDone = false; //Продолжаем пока хотя бы одна область нуждается в обработке
                        break;
                    }
                }

                var chunkResultsArray = new int[areas.Length];
                resultBuffer.GetData(chunkResultsArray);
                chunkResultsList.Add(chunkResultsArray);

                if (allAreasDone)
                    break;
            }
            //for (int i = 0; i < chunksNumber; i++)
            //{
            //    _computeShader.SetInt("SpriteStartIndex", i * span);
            //    _computeShader.SetInt("SpriteEndIndex", (i + 1) * span);

            //    _computeShader.Dispatch(algorythmKernel, iterationsCount, 1, 1);

            //    var chunkResultsArray = new int[areas.Length];
            //    resultBuffer.GetData(chunkResultsArray);
            //    chunkResultsList.Add(chunkResultsArray);
            //}

            stopWatch.Stop();
            var ts = stopWatch.Elapsed;
            Debug.Log($"Диспатч прошел. Занял он {ts}. passes = {passes}");

            var resultData = new int[areas.Length];
            for (int i = 0; i < resultData.Length; i++)
            {
                var totalScore = 0;
                for (int j = 0; j < chunkResultsList.Count; j++)
                    totalScore += chunkResultsList[j][i];
                resultData[i] = totalScore;
            }


            // 3а. Проверяем не надурили ли мы с алгоритмом

            var scoresList = new List<int>();
            var testValues1List = new List<int>();
            var testValues2List = new List<int>();

            for (int m = 0; m < /*areasList.Count*/100; m++)
            {
                //var i = j;
                var (count, testValue1, testValue2) = countScoreForArea(areasList[m]);
                var c = count;
                var ar = areasList[m];
                var s = (int)ar.Score;
                if (s == 0 || c == 0)
                {

                }
                scoresList.Add(c * s);
                testValues1List.Add(testValue1);
                //testValues2List.Add(testValue2);
            }

            var resultIsCorrect = true;
            for (int n = 0; n < scoresList.Count; n++)
            {
                //var s = (int)areasList[i].Score;
                var a = resultData[n];
                var b = scoresList[n];
                Debug.Log($"checking id {a} == {b}");
                if (a != b)
                {
                    resultIsCorrect = false;
                    break;
                }
            }

            Debug.Log($"Результат проверен. Он {resultIsCorrect}.");
            return result;
            Debug.Log($"3");
            Thread.Sleep(1000);


            // 4. Выясняем победителя и забираем его результат из данных

            var maxScoreIndex = 0;
            var maxScore = -1;
            for (int i = 0; i < resultData.Length; i++)
            {
                if (resultData[i] > maxScore)
                {
                    maxScore = resultData[i];
                    maxScoreIndex = i;
                }
            }

            for (int u = 0; u < 100; u++)
            {
                var score = resultData[u];
            }

            Debug.Log($"maxScoreIndex = {maxScoreIndex}, maxScore = {maxScore}");
            var theWinnerArea = areasList[maxScoreIndex];
            theWinnerArea.Selected = true;

            var pixelsRemoved = 0;
            Parallel.For(0, _sprites.Length, (spriteIndex, loopState) =>
            {
                var sprite = _sprites[spriteIndex];
                var lastSpriteX = sprite.Length - theWinnerArea.Dimensions.X + 1;
                var lastSpriteY = sprite[0].Length - theWinnerArea.Dimensions.Y + 1;

                for (int spriteX = 0; spriteX < lastSpriteX; spriteX++)
                {
                    for (int spriteY = 0; spriteY < lastSpriteY; spriteY++)
                    {
                        var maybeThis = true;
                        for (int areaX = 0; areaX < theWinnerArea.Dimensions.X; areaX++)
                        {
                            for (int areaY = 0; areaY < theWinnerArea.Dimensions.Y; areaY++)
                            {
                                var pixelX = spriteX + areaX;
                                var pixelY = spriteY + areaY;
                                var candidatePixel = sprite[pixelX][pixelY];

                                var areaPixel = _sprites[theWinnerArea.SpriteIndex][theWinnerArea.SpriteRect.X + areaX][theWinnerArea.SpriteRect.Y + areaY];
                                if (areaPixel.Color != candidatePixel.Color)
                                {
                                    maybeThis = false;
                                    break;
                                }
                            }

                            if (!maybeThis)
                                break;
                        }

                        if (maybeThis)
                        {
                            var newCorrelation = new MyAreaCoordinates(spriteIndex, spriteX, spriteY, theWinnerArea.Dimensions.X, theWinnerArea.Dimensions.Y);
                            MyArea.EraseAreaFromSprite(sprite, spriteX, spriteY, theWinnerArea.Dimensions);
                            pixelsRemoved += theWinnerArea.OpaquePixelsCount;
                            theWinnerArea.CorrelationsBag.Add(newCorrelation);
                        }
                    }
                }
            });

            UnprocessedPixels -= pixelsRemoved;

            Debug.Log($"Удалено: {pixelsRemoved}. Осталось {UnprocessedPixels}");

            Debug.Log($"4");
            Thread.Sleep(1000);


            // 5. Обновляем данные и области

            dataOffset = 0;
            for (int i = 0; i < _sprites.Length; i++)
            {
                var sprite = _sprites[i];
                var width = sprite.Length;
                var height = sprite[0].Length;

                for (int x = 0; x < width; x++)
                    for (int y = 0; y < height; y++)
                        data[dataOffset++] = sprite[x][y].R << 24 | sprite[x][y].G << 16 | sprite[x][y].B << 8 | sprite[x][y].A;
            }

            for (int i = 0; i < areasList.Count; i++)
            {
                var area = areasList[i];
                var mask = area.Selected ? 0 : 1;
                var metaAndSpriteIndex = mask << 24 | area.SpriteIndex & 16777215;
                areas[i].MetaAndSpriteIndex = metaAndSpriteIndex;
                //areas[i] = new areaStruct(metaAndSpriteIndex, area.SpriteRect.X << 16 | area.SpriteRect.Y, area.SpriteRect.Width << 16 | area.SpriteRect.Height);
            }

            Debug.Log($"5");
            Thread.Sleep(1000);

            break;
        }

        //Debug.LogError($"areas.Length = {areas.Length}");
        //Debug.LogError($"registry.Length = {registry.Length}");
        //Debug.LogError($"data.Length = {data.Length}");

        //Debug.LogError($"_allAreas.Count() = {_allAreas.Count()}");

        //for (int i = 0; i < 100; i++)
        //{
        //    Debug.Log($"area #{i}: ops count - {counts[i]}");
        //}

        Debug.Log($"Забрали результат");

        dataBuffer.Dispose();
        registryBuffer.Dispose();
        areasBuffer.Dispose();
        resultBuffer.Dispose();

        //var scoresList = new List<int>();
        //var testValues1List = new List<int>();
        //var testValues2List = new List<int>();

        //for (int m = 0; m < /*areasList.Count*/100; m++)
        //{
        //    //var i = j;
        //    var (count, testValue1, testValue2) = countScoreForArea(areasList[m]);
        //    var c = count;
        //    var ar = areasList[m];
        //    var s = (int)ar.Score;
        //    if (s == 0 || c == 0)
        //    {

        //    }
        //    scoresList.Add(c * s);
        //    //testValues1List.Add(testValue1);
        //    //testValues2List.Add(testValue2);
        //}

        //var resultIsCorrect = true;
        //for (int n = 0; n < scoresList.Count; n++)
        //{
        //    //var s = (int)areasList[i].Score;
        //    var a = resultData[n];
        //    var b = scoresList[n];
        //    Debug.Log($"checking id {a} == {b}");
        //    if (a != b)
        //    {
        //        resultIsCorrect = false;
        //        break;
        //    }
        //}

        //Debug.Log($"Результат проверен. Он {resultIsCorrect}.");

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
            var lastSpriteX = sprite.Length - area.Dimensions.X + 1;
            var lastSpriteY = sprite[0].Length - area.Dimensions.Y + 1;

            //testValue2 += area.SpriteRect.X + area.SpriteRect.Y + area.SpriteRect.Width + area.SpriteRect.Height;

            for (int spriteX = 0; spriteX < lastSpriteX; spriteX++)
            {
                for (int spriteY = 0; spriteY < lastSpriteY; spriteY++)
                {
                    testValue1++;
                    var maybeThis = true;
                    //var atLeastOneOpaque = false;
                    for (int areaX = 0; areaX < area.Dimensions.X; areaX++)
                    {
                        for (int areaY = 0; areaY < area.Dimensions.Y; areaY++)
                        {
                            var pixelX = spriteX + areaX;
                            var pixelY = spriteY + areaY;
                            var candidatePixel = sprite[pixelX][pixelY];

                            var areaPixel = _sprites[area.SpriteIndex][area.SpriteRect.X + areaX][area.SpriteRect.Y + areaY];
                            //testValue1 += areaPixel.Color;
                            //testValue2 += candidatePixel.Color;

                            if (areaPixel.A == 0 && candidatePixel.A == 0)
                                continue;

                            //atLeastOneOpaque = true;
                            if (areaPixel.Color != candidatePixel.Color)
                            {
                                maybeThis = false;
                                break;
                            }
                            //if (areaPixel.R != candidatePixel.R || areaPixel.G != candidatePixel.G || areaPixel.B != candidatePixel.B || areaPixel.A != candidatePixel.A)
                            //{
                            //    maybeThis = false;
                            //    break;
                            //}
                        }

                        if (!maybeThis)
                            break;
                    }

                    if (maybeThis/* && atLeastOneOpaque*/)
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

        List<KeyValuePair<string, MyArea>> orderedAreas = null;
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

            //Debug.Log($"Removing area #{currentAreaIndex}: hash {currentArea.Key}, score {currentArea.Value.Correlations.Count * currentArea.Value.Score}");
            //var areasRemoved = await applyBestArea(_sprites, currentArea.Key);
            //var pixelsRemoved = currentArea.Value.OpaquePixelsCount * areasRemoved.Count;
            //map.Add(currentArea.Value, areasRemoved);
            //OverallProgressReport.OperationsDone += pixelsRemoved;
            //UnprocessedPixels -= pixelsRemoved;
            //currentAreaIndex++;
            //Debug.LogError($"Run 6");
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

    //private async Task<List<MyAreaCoordinates>> applyBestArea(MyColor[][][] sprites, int bestAreaIndex)
    //{
    //    var result = new List<MyAreaCoordinates>();
    //    MyArea bestArea;
    //    if (!_allAreas.TryRemove(bestAreaIndex, out bestArea))
    //        throw new ApplicationException($"Area is not found!");

    //    var correlations = bestArea.Correlations;
    //    foreach (var kvp in correlations)
    //    {
    //        var myAreaCoordinates = kvp.Value;
    //        if (_mapOfEmptiness.Contains(myAreaCoordinates.Dimensions, myAreaCoordinates.SpriteIndex, myAreaCoordinates.X, myAreaCoordinates.Y))
    //            continue;

    //        var candidateForErasing = MyArea.CreateFromSprite(sprites[myAreaCoordinates.SpriteIndex], myAreaCoordinates.SpriteIndex, myAreaCoordinates.X, myAreaCoordinates.Y, myAreaCoordinates.Dimensions);
    //        if (candidateForErasing.GetHashCode() != bestArea.GetHashCode())
    //            continue;
    //        MyArea.EraseAreaFromSprite(sprites[myAreaCoordinates.SpriteIndex], myAreaCoordinates.X, myAreaCoordinates.Y, myAreaCoordinates.Dimensions);
    //        _mapOfEmptiness.MakeEmpty(myAreaCoordinates.Dimensions, myAreaCoordinates.SpriteIndex, myAreaCoordinates.X, myAreaCoordinates.Y);
    //        result.Add(myAreaCoordinates);
    //    }

    //    return result;
    //}
}
