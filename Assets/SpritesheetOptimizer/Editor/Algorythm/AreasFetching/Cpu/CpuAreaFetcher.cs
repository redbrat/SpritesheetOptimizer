using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class CpuAreaFetcher : IAreaFetcher
{
    private readonly CancellationToken _ct;
    private readonly MapOfEmptiness _mapOfEmptiness;

    public CpuAreaFetcher(CancellationToken ct, MapOfEmptiness mapOfEmptiness)
    {
        _ct = ct;
        _mapOfEmptiness = mapOfEmptiness;
    }

    public Task<ConcurrentDictionary<int, MyArea>> FetchAreas(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var result = Task.Run(() => getAllAreas(sprites, areaSizings, areaEnumerator, progressReport));
        return result;
    }
    private ConcurrentDictionary<int, MyArea> getAllAreas(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var areas = new ConcurrentDictionary<int, MyArea>();

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
                var spritesAreas = getUniqueAreas(targetSizing, spriteIndex, areas, areaEnumerator, progressReport);
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
        return areas;
    }

    /// <returns>(Overall areas, Unique areas)</returns>
    private (int total, int unique) getUniqueAreas(MyVector2 areaSizing, int spriteIndex, ConcurrentDictionary<int, MyArea> areas, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        var areasTotal = 0;
        var areasUnique = 0;
        areaEnumerator .EnumerateThroughSprite(areaSizing, spriteIndex, (sprite, index, x, y) =>
        {
            if (!_mapOfEmptiness.Contains(areaSizing, index, x, y))
            {
                var area = MyArea.CreateFromSprite(sprite, x, y, areaSizing);
                var hash = area.GetHashCode();
                if (areas.TryAdd(hash, area))
                    areasUnique++;
                area = areas[hash];

                area.Correlations.TryAdd(area.Correlations.Count, new MyAreaCoordinates(index, x, y, areaSizing.X, areaSizing.Y));

                areasTotal++;
            }
        });
        progressReport.OperationsDone++;
        return (areasTotal, areasUnique);
    }
}
