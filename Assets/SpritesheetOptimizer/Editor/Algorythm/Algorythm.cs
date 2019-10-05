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

    private struct taskStruct
    {
        public int MetaAndSpriteIndex;
        public int SpriteXAndY;
        public int XAndY;

        public taskStruct(int metaAndSpriteIndex, int spriteXAndY, int xAndY)
        {
            MetaAndSpriteIndex = metaAndSpriteIndex;
            SpriteXAndY = spriteXAndY;
            XAndY = xAndY;
        }
    }

    public async Task<List<MyArea>> Run()
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
                tasks[i] = new taskStruct(metaAndSpriteIndex, 0, 0);
        }
        var tasksBuffer = new ComputeBuffer(tasks.Length, 12);
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
            while(true)
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
                var lastSpriteX = sprite.Length - theWinnerArea.Dimensions.X;
                var lastSpriteY = sprite[0].Length - theWinnerArea.Dimensions.Y;

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
            var lastSpriteX = sprite.Length - area.Dimensions.X;
            var lastSpriteY = sprite[0].Length - area.Dimensions.Y;

            //testValue2 += area.SpriteRect.X + area.SpriteRect.Y + area.SpriteRect.Width + area.SpriteRect.Height;

            for (int spriteX = 0; spriteX < lastSpriteX; spriteX++)
            {
                for (int spriteY = 0; spriteY < lastSpriteY; spriteY++)
                {
                    testValue1++;
                    var maybeThis = true;
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
