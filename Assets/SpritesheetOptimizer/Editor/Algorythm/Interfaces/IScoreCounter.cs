using System.Collections.Generic;

public interface IScoreCounter
{
    IEnumerable<(int, MyArea)> CountScores(IEnumerable<(int, MyArea)> result, MyColor[][][] sprites, IEnumerable<MyArea> allAreas, IAreaEnumerator areaEnumerator, ProgressReport progressReport);
}