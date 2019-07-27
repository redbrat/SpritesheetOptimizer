using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
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

    private IEnumerable<MyVector2> _areaSizings;
    private IEnumerable<MyArea> _allAreas;
    private IAreaEnumerator _areaEnumerator;

    public Algorythm(MyColor[][][] sprites, Type areaEnumeratorType, IList<ISizingsConfigurator> sizingConfigurators, IList<IScoreCounter> scoreCounters)
    {
        ProgressReport = new ProgressReport();
        _sprites = sprites;
        _areaEnumeratorType = areaEnumeratorType;
        _sizingsConfigurators = sizingConfigurators;
        _scoreCounters = scoreCounters;
    }

    public async Task Initialize(Vector2Int maxAreaSize)
    {
        _areaSizings = getAreaSizings(_sprites, maxAreaSize);

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

    private IEnumerable<MyVector2> getAreaSizings(MyColor[][][] sprites, Vector2Int maxAreaSize)
    {
        var result = default(IEnumerable<MyVector2>);
        for (int i = 0; i < _sizingsConfigurators.Count; i++)
            result = _sizingsConfigurators[i]?.ConfigureSizings(result, sprites.Length, maxAreaSize.x, maxAreaSize.y);
        return result;
    }

    private int countUprocessedPixels(MyVector2 areaSizing, IAreaEnumerator areaEnumerator)
    {
        var result = 0;
        areaEnumerator.Enumerate(areaSizing, (sprite, x, y) =>
        {
            if (sprite[x][y].A > 0)
                result++;
        });
        return result;
    }

    private IEnumerable<MyArea> getAllAreas(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var overallOpsCount = areaSizings.Count() * sprites.Length;

        progressReport.OperationDescription = "Fetching possible areas";
        progressReport.OperationsCount = overallOpsCount;
        progressReport.OperationsDone = 0;

        var sizingsList = areaSizings.ToList();
        var areas = new ConcurrentDictionary<int, MyArea>();

        try
        {
            Parallel.For(0, overallOpsCount, (int index, ParallelLoopState state) =>
            {
                if (state.IsExceptional)
                    Debug.Log("Exception!");
                var areaVariantIndex = Mathf.FloorToInt(index / sprites.Length);
                var spriteIndex = index - areaVariantIndex * sprites.Length;
                if (areaVariantIndex > sizingsList.Count - 1)
                    Debug.LogError($"sizingsList[areaVariantIndex] is out of range! areaVariantIndex = {areaVariantIndex}, sizingsList.Count = {sizingsList.Count}");
                if (areaVariantIndex < 0)
                    Debug.LogError($"areaVariantIndex < 0! ({areaVariantIndex})");
                var targetArea = sizingsList[areaVariantIndex];
                getUniqueAreas(targetArea, spriteIndex, areas, areaEnumerator, progressReport);
            });
        }
        catch (AggregateException ae)
        {
            Debug.Log("catch");
            ae.Handle((inner) =>
            {
                Debug.Log($"{inner.Message}\r\n\r\n{inner.StackTrace}");
                return true;
            });
        }

        return areas.Values;
    }

    private void getUniqueAreas(MyVector2 areaSizing, int spriteIndex, ConcurrentDictionary<int, MyArea> areas, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        areaEnumerator.EnumerateThroughSprite(areaSizing, spriteIndex, (sprite, x, y) =>
        {
            var area = MyArea.CreateFromSprite(sprite, x, y, areaSizing);
            var hash = area.GetHashCode();
            areas.TryAdd(hash, area);
        });
        progressReport.OperationsDone++;
    }

    #endregion Initializing

    public async Task<Dictionary<MyArea, List<(int, int, int)>>> Run()
    {
        var map = new Dictionary<MyArea, List<(int, int, int)>>();

        while (UnprocessedPixels > 0)
        {
            Debug.Log("1");
            var scores = await Task.Run(() => countScoreForEachArea(_sprites, _allAreas, _areaEnumerator));

            Debug.Log("2");
            var rating = scores.OrderByDescending(kvp => kvp.Item1);

            Debug.Log("3");
            var bestArea = getBestArea(rating);

            Debug.Log("4");
            UnprocessedPixels -= applyBestArea(_sprites, bestArea, map);
        }

        return map;
    }

    private IEnumerable<(int, MyArea)> countScoreForEachArea(MyColor[][][] sprites, IEnumerable<MyArea> allAreas, IAreaEnumerator areaEnumerator)
    {
        var result = default(IEnumerable<(int, MyArea)>);
        foreach (var scoreCounter in _scoreCounters)
            result = scoreCounter.CountScores(result, sprites, allAreas, areaEnumerator, ProgressReport);
        return result;
    }

    private MyArea getBestArea(IOrderedEnumerable<(int, MyArea)> rating) => rating.First().Item2;

    private int applyBestArea(MyColor[][][] sprites, MyArea bestArea, Dictionary<MyArea, List<(int, int, int)>> map)
    {
        var result = 0;
        var winnerAreaDimensions = bestArea.Dimensions;
        var mappedAreas = new List<(int, int, int)>();

        for (int i = 0; i < sprites.Length; i++)
        {
            var sprite = sprites[i];
            for (int x = 0; x < sprite.Length - winnerAreaDimensions.X; x++)
            {
                for (int y = 0; y < sprite[x].Length - winnerAreaDimensions.Y; y++)
                {
                    var comparedArea = MyArea.CreateFromSprite(sprite, x, y, winnerAreaDimensions);
                    if (comparedArea.GetHashCode() == bestArea.GetHashCode())
                    {
                        MyArea.EraseAreaFromSprite(sprite, x, y, winnerAreaDimensions);

                        mappedAreas.Add((i, x, y));
                        result += bestArea.OpaquePixelsCount;
                    }
                }
            }
        }

        map.Add(bestArea, mappedAreas);
        return result;
    }
}
