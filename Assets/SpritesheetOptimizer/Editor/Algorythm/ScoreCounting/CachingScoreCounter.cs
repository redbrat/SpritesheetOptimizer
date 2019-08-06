using System.Collections.Generic;
using System.Threading;

public class CachingScoreCounter : DefaultScoreCounter
{
    private IEnumerable<(int, MyArea)> _cache;

    public override IEnumerable<(int, MyArea)> CountScores(IEnumerable<(int, MyArea)> result, MyColor[][][] sprites, IEnumerable<MyArea> allAreas, IAreaEnumerator areaEnumerator, ProgressReport progressReport, CancellationToken ct)
    {
        if (_cache == null)
            _cache = base.CountScores(result, sprites, allAreas, areaEnumerator, progressReport, ct);
        return _cache;
    }
}
