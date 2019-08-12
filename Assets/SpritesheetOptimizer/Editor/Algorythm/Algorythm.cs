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
    }

    public async Task Initialize(Vector2Int maxAreaSize, CancellationToken ct)
    {
        _ct = ct;
        _areaSizings = getAreaSizings(_sprites, maxAreaSize, ct);

        var areaEnumeratorCtor = _areaEnumeratorType.GetConstructor(new Type[] { typeof(MyColor[][][]) });
        if (areaEnumeratorCtor == null)
            areaEnumeratorCtor = _areaEnumeratorType.GetConstructor(new Type[] { typeof(MyColor[][][]), typeof(IEnumerable<MyVector2>) });
        else
            _areaEnumerator = (IAreaEnumerator)areaEnumeratorCtor.Invoke(new object[] { _sprites });
        if (_areaEnumerator == null && areaEnumeratorCtor == null)
            throw new ArgumentException($"Got AreaEnumerator with unknown set of ctor parameters.");
        else if (_areaEnumerator == null)
            _areaEnumerator = (IAreaEnumerator)areaEnumeratorCtor.Invoke(new object[] { _sprites, _areaSizings });
        UnprocessedPixels = countUprocessedPixels(MyVector2.One, _areaEnumerator);
        _mapOfEmptiness = new MapOfEmptiness();
        await _mapOfEmptiness.Initialize(_areaSizings, _sprites, _areaEnumerator);

        switch (_computeMode)
        {
            case ComputeMode.Cpu:
                _areaFetcher = new CpuAreaFetcher(_ct, _mapOfEmptiness);
                break;
            case ComputeMode.Gpu:
                _areaFetcher = new GpuAreaFetcher();
                break;
        }
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

    public async Task<Dictionary<MyArea, List<MyAreaCoordinates>>> Run()
    {
        OverallProgressReport.OperationDescription = "Removing areas from picture";
        OverallProgressReport.OperationsCount = UnprocessedPixels;
        var map = new Dictionary<MyArea, List<MyAreaCoordinates>>();

        var currentAreaIndex = 0;

        List<KeyValuePair<int, MyArea>> orderedAreas = null;
        while (UnprocessedPixels > 0)
        {
            if (_ct.IsCancellationRequested)
                break;
            if (currentAreaIndex % _areasFreshmentSpan == 0)
                await setAreasAndScores();

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
                            var correlatedArea = MyArea.CreateFromSprite(sprite, correlation.X, correlation.Y, correlation.Dimensions);
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
        }
        Debug.Log($"Done!");

        return map;
    }

    private async Task setAreasAndScores()
    {
        _allAreas = await _areaFetcher.FetchAreas(_sprites, _areaSizings, _areaEnumerator, OperationProgressReport);
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

            var candidateForErasing = MyArea.CreateFromSprite(sprites[myAreaCoordinates.SpriteIndex], myAreaCoordinates.X, myAreaCoordinates.Y, myAreaCoordinates.Dimensions);
            if (candidateForErasing.GetHashCode() != bestArea.GetHashCode())
                continue;
            MyArea.EraseAreaFromSprite(sprites[myAreaCoordinates.SpriteIndex], myAreaCoordinates.X, myAreaCoordinates.Y, myAreaCoordinates.Dimensions);
            _mapOfEmptiness.MakeEmpty(myAreaCoordinates.Dimensions, myAreaCoordinates.SpriteIndex, myAreaCoordinates.X, myAreaCoordinates.Y);
            result.Add(myAreaCoordinates);
        }

        return result;
    }
}
