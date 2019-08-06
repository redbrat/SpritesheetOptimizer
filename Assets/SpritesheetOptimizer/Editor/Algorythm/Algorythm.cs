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

    private readonly MyColor[][][] _sprites;
    private readonly IList<ISizingsConfigurator> _sizingsConfigurators;
    private readonly IList<IScoreCounter> _scoreCounters;
    private readonly Type _areaEnumeratorType;

    private ConcurrentDictionary<MyArea, long> _allAreas;
    private IEnumerable<MyVector2> _areaSizings;
    private IAreaEnumerator _areaEnumerator;

    private CancellationToken _ct;

    public Algorythm(MyColor[][][] sprites, Type areaEnumeratorType, IList<ISizingsConfigurator> sizingConfigurators, IList<IScoreCounter> scoreCounters)
    {
        ProgressReport = new ProgressReport();
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
        _allAreas = await Task.Run(() => getAllAreas(_sprites, _areaSizings, _areaEnumerator, ProgressReport));
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

    private ConcurrentDictionary<MyArea, long> getAllAreas(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var result = new ConcurrentDictionary<MyArea, long>();

        var overallOpsCount = areaSizings.Count() * sprites.Length;

        progressReport.OperationDescription = "Fetching possible areas";
        progressReport.OperationsCount = overallOpsCount;
        progressReport.OperationsDone = 0;

        var sizingsList = areaSizings.ToList();

        var allAreas = 0;
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
                var spritesAreas = getUniqueAreas(targetSizing, spriteIndex, result, areaEnumerator, progressReport);
                //Debug.Log($"allAreas before = {allAreas}");
                allAreas += spritesAreas;
                //Debug.Log($"allAreas after = {allAreas}");
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
        Debug.Log($"allAreas = {allAreas}, areas.Count = {result.Count}");
        return result;
    }

    /// <returns>Overall areas</returns>
    private int getUniqueAreas(MyVector2 areaSizing, int spriteIndex, ConcurrentDictionary<MyArea, long> areas, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var result = 0;
        areaEnumerator.EnumerateThroughSprite(areaSizing, spriteIndex, (sprite, index, x, y) =>
        {
            var area = MyArea.CreateFromSprite(sprite, x, y, areaSizing);
            //var hash = area.GetHashCode();
            //areas.TryAdd(hash, area);

            var dirtyScore = (long)(Mathf.Pow(area.OpaquePixelsCount, 3f) / area.Dimensions.Square);
            areas.AddOrUpdate(area, dirtyScore, (exisitingKey, existingScore) => existingScore + dirtyScore);

            result++;
        });
        progressReport.OperationsDone++;
        return result;
    }

    #endregion Initializing

    public async Task<Dictionary<MyArea, List<(int, int, int)>>> Run()
    {
        var map = new Dictionary<MyArea, List<(int, int, int)>>();

        Debug.Log("0");
        while (UnprocessedPixels > 0)
        {
            if (_ct.IsCancellationRequested)
                break;
            //Debug.Log("1");
            //var scores = await Task.Run(() => countScoreForEachArea(_sprites, _allAreas, _areaEnumerator));

            Debug.Log("2");
            var rating = _allAreas.OrderByDescending(kvp => kvp.Key);

            Debug.Log("3");
            var bestArea = getBestArea(rating);
            Debug.Log("4");

            if (_ct.IsCancellationRequested)
                break;
            Debug.Log("5");
            UnprocessedPixels -= applyBestArea(_sprites, _areaEnumerator, bestArea, map, _ct);
            Debug.Log("6");
        }

        return map;
    }

    //private IEnumerable<(int, MyArea)> countScoreForEachArea(MyColor[][][] sprites, IEnumerable<MyArea> allAreas, IAreaEnumerator areaEnumerator)
    //{
    //    var result = default(IEnumerable<(int, MyArea)>);
    //    foreach (var scoreCounter in _scoreCounters)
    //        result = scoreCounter.CountScores(result, sprites, allAreas, areaEnumerator, ProgressReport, _ct);
    //    return result;
    //}

    private MyArea getBestArea(IOrderedEnumerable<KeyValuePair<MyArea, long>> rating) => rating.First().Key;

    private int applyBestArea(MyColor[][][] sprites, IAreaEnumerator enumerator, MyArea bestArea, Dictionary<MyArea, List<(int, int, int)>> map, CancellationToken ct)
    {
        var result = 0;
        var mappedAreas = new List<(int, int, int)>();
        var winnerAreaDimensions = bestArea.Dimensions;

        enumerator.EnumerateParallel(bestArea.Dimensions, (sprite, spriteIndex, x, y) =>
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

        long value;
        _allAreas.TryRemove(bestArea, out value);

        return result;
    }
}
