using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class Algorythm
{
    public readonly ProgressReport ProgressReport;

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

    private ConcurrentDictionary<int, MyArea> _allAreas;
    //private IOrderedEnumerable<KeyValuePair<int, long>> _allScores;
    private IEnumerable<MyVector2> _areaSizings;
    private IAreaEnumerator _areaEnumerator;

    private CancellationToken _ct;

    public Algorythm(MyColor[][][] sprites, Type areaEnumeratorType, IList<ISizingsConfigurator> sizingConfigurators, IList<IScoreCounter> scoreCounters, int areasFreshmentSpan, int areasVolatilityRange)
    {
        ProgressReport = new ProgressReport();

        _areasFreshmentSpan = areasFreshmentSpan;
        _areasVolatilityRange = areasVolatilityRange;

        _sprites = sprites;
        _areaEnumeratorType = areaEnumeratorType;
        _sizingsConfigurators = sizingConfigurators;
        _scoreCounters = scoreCounters;
    }

    public async Task Initialize(Vector2Int maxAreaSize, CancellationToken ct)
    {
        _ct = ct;
        _areaSizings = getAreaSizings(_sprites, maxAreaSize, ct);

        var areaEnumeratorCtor = _areaEnumeratorType.GetConstructor(new Type[] { typeof(MyColor[][][]) });
        foreach (var param in areaEnumeratorCtor.GetParameters())
        {
            Debug.Log($"ParameterType = {param.ParameterType}");
        }
        Debug.Log($"_sprites = {_sprites}. Type = {_sprites.GetType()}");
        if (areaEnumeratorCtor == null)
            areaEnumeratorCtor = _areaEnumeratorType.GetConstructor(new Type[] { typeof(MyColor[][][]), typeof(IEnumerable<MyVector2>) });
        else
            _areaEnumerator = (IAreaEnumerator)areaEnumeratorCtor.Invoke(new object[] { _sprites });
        //_areaEnumerator = (IAreaEnumerator)Activator.CreateInstance(_areaEnumeratorType, new object[] { _sprites });
        if (_areaEnumerator == null && areaEnumeratorCtor == null)
            throw new ArgumentException($"Got AreaEnumerator with unknown set of ctor parameters.");
        else if (_areaEnumerator == null)
            _areaEnumerator = (IAreaEnumerator)areaEnumeratorCtor.Invoke(new object[] { _sprites, _areaSizings });
        //_areaEnumerator = (IAreaEnumerator)Activator.CreateInstance(_areaEnumeratorType, _sprites, _areaSizings);
        UnprocessedPixels = countUprocessedPixels(MyVector2.One, _areaEnumerator);
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

    public async Task<Dictionary<MyArea, List<(int, int, int)>>> Run()
    {
        var map = new Dictionary<MyArea, List<(int, int, int)>>();

        //var areasOrderedByScores = _allAreas.OrderByDescending(kvp => kvp.Value.Correlations.Count * kvp.Value.Score).ToList();

        var currentAreaIndex = 0;
        var initialUnprocesedPixels = UnprocessedPixels;
        var processedPixels = 0;

        List<KeyValuePair<int, MyArea>> orderedAreas = null;
        while (UnprocessedPixels > 0)
        {
            if (_ct.IsCancellationRequested)
                break;
            if (currentAreaIndex % _areasFreshmentSpan == 0)
            {
                Debug.Log($"Areas and scores recounting...");
                await setAreasAndScores();
            }
            ProgressReport.OperationDescription = "Removing areas from picture";
            ProgressReport.OperationsCount = initialUnprocesedPixels;

            //После удаления некоторых пикселей рейтинги областей могут меняться - поэтому надо обновлять и переупорядочивать каждый раз.
            if (currentAreaIndex > 0)
            {
                for (int i = 0; i < _areasVolatilityRange; i++)
                {
                    var area = orderedAreas[i].Value;
                    var invalidAreas = new List<int>();
                    foreach (var kvp in area.Correlations)
                    {
                        var correlation = kvp.Value;
                        var sprite = _sprites[correlation.SpriteIndex];
                        var correlatedArea = MyArea.CreateFromSprite(sprite, correlation.X, correlation.Y, correlation.Dimensions);
                        if (correlatedArea.GetHashCode() != area.GetHashCode())
                            invalidAreas.Add(kvp.Key);
                    }
                    for (int j = 0; j < invalidAreas.Count; j++)
                    {
                        MyAreaCoordinates val;
                        area.Correlations.TryRemove(invalidAreas[j], out val);
                    }
                    //for (int j = 0; j < area.Correlations.Count; j++)
                    //{
                    //    var correlation = area.Correlations.[j];
                    //    var sprite = _sprites[correlation.SpriteIndex];
                    //    var correlatedArea = MyArea.CreateFromSprite(sprite, correlation.X, correlation.Y, correlation.Dimensions);
                    //    if (correlatedArea.GetHashCode() != area.GetHashCode())
                    //    {
                    //        area.Correlations.RemoveAt(j);
                    //        j--;
                    //    }
                    //}
                }
            }
            orderedAreas = _allAreas.OrderByDescending(kvp => kvp.Value.Correlations.Count * kvp.Value.Score).ToList();
            var currentArea = orderedAreas[0];

            Debug.Log($"Working with score #{currentAreaIndex}: id {currentArea.Key}, score {currentArea.Value.Correlations.Count * currentArea.Value.Score}");
            var pixelsRemoved = await applyBestArea(_sprites, _areaEnumerator, currentArea.Key, map, _ct);
            processedPixels += pixelsRemoved;
            ProgressReport.OperationsDone = processedPixels;
            UnprocessedPixels -= pixelsRemoved;
            currentAreaIndex++;
        }
        Debug.Log($"Done!");

        return map;
    }

    private async Task setAreasAndScores()
    {
        _allAreas = await Task.Run(() => getAllAreas(_sprites, _areaSizings, _areaEnumerator, ProgressReport));
        //var (allAreas, allScores) = await Task.Run(() => getAllAreas(_sprites, _areaSizings, _areaEnumerator, ProgressReport));
        //_allAreas = allAreas.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        //_allScores = allScores.OrderByDescending(kvp => kvp.Value);
    }

    private /*(ConcurrentDictionary<int, MyArea> areas, ConcurrentDictionary<int, long> scores)*/ConcurrentDictionary<int, MyArea> getAllAreas(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var areas = new ConcurrentDictionary<int, MyArea>();
        //var scores = new ConcurrentDictionary<int, long>();

        var overallOpsCount = areaSizings.Count() * sprites.Length;

        progressReport.OperationDescription = "Fetching possible areas";
        progressReport.OperationsCount = overallOpsCount;
        progressReport.OperationsDone = 0;

        var sizingsList = areaSizings.ToList();

        var allAreas = 0;
        var uniqueAreas = 0;
        try
        {
            Parallel.For(0, overallOpsCount, (int index, ParallelLoopState state) =>
            {
                if (state.IsExceptional)
                    Debug.Log("Exception!");
                if (_ct.IsCancellationRequested)
                    state.Break();
                var sizingIndex = Mathf.FloorToInt(index / sprites.Length);
                var spriteIndex = index - sizingIndex * sprites.Length;
                if (sizingIndex > sizingsList.Count - 1)
                    Debug.LogError($"sizingsList[sizingIndex] is out of range! sizingIndex = {sizingIndex}, sizingsList.Count = {sizingsList.Count}");
                if (sizingIndex < 0)
                    Debug.LogError($"sizingIndex < 0! ({sizingIndex})");
                var targetSizing = sizingsList[sizingIndex];
                var spritesAreas = getUniqueAreas(targetSizing, spriteIndex, areas/*, scores*/, areaEnumerator, progressReport);
                allAreas += spritesAreas.total;
                uniqueAreas += spritesAreas.unique;
            });
        }
        catch (AggregateException ae)
        {
            Debug.Log("catch");
            ae.Handle((inner) =>
            {
                Debug.LogError($"{inner.Message}\r\n\r\n{inner.StackTrace}");
                return true;
            });
        }
        //Debug.Log($"areas = {areas}, areas.Count = {areas.Count}, scores = {scores}, scores.Count = {scores.Count}");
        return /*(areas, scores)*/ areas;
    }

    /// <returns>(Overall areas, Unique areas)</returns>
    private (int total, int unique) getUniqueAreas(MyVector2 areaSizing, int spriteIndex, ConcurrentDictionary<int, MyArea> areas, /*ConcurrentDictionary<int, long> scores, */IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var areasTotal = 0;
        var areasUnique = 0;
        areaEnumerator.EnumerateThroughSprite(areaSizing, spriteIndex, (sprite, index, x, y) =>
        {
            var area = MyArea.CreateFromSprite(sprite, x, y, areaSizing);
            var hash = area.GetHashCode();
            if (areas.TryAdd(hash, area))
                areasUnique++;
            area = areas[hash];

            //var dirtyScore = (long)(Mathf.Pow(area.OpaquePixelsCount, 3f) / area.Dimensions.Square);
            //scores.AddOrUpdate(hash, area.Score, (exisitingKey, existingScore) => existingScore + area.Score);
            area.Correlations.TryAdd(area.Correlations.Count, new MyAreaCoordinates(index, x, y, areaSizing.X, areaSizing.Y));

            areasTotal++;
        });
        progressReport.OperationsDone++;
        return (areasTotal, areasUnique);
    }

    private int getBestArea(IOrderedEnumerable<KeyValuePair<int, long>> rating) => rating.First().Key;

    private async Task<int> applyBestArea(MyColor[][][] sprites, IAreaEnumerator enumerator, int areaIndex, Dictionary<MyArea, List<(int, int, int)>> map, CancellationToken ct)
    {
        var result = 0;
        var mappedAreas = new List<(int, int, int)>();
        MyArea bestArea;
        if (!_allAreas.TryRemove(areaIndex, out bestArea))
            throw new ApplicationException($"Area is not found!");
        var winnerAreaDimensions = bestArea.Dimensions;

        await enumerator.EnumerateParallel(bestArea.Dimensions, (sprite, spriteIndex, x, y) =>
        {
            var comparedArea = MyArea.CreateFromSprite(sprite, x, y, winnerAreaDimensions);
            if (comparedArea.GetHashCode() == bestArea.GetHashCode())
            {
                MyArea.EraseAreaFromSprite(sprite, x, y, winnerAreaDimensions);

                mappedAreas.Add((spriteIndex, x, y));
                result += bestArea.OpaquePixelsCount;
            }
        }, ct);

        map.Add(bestArea, mappedAreas);

        return result;
    }
}
