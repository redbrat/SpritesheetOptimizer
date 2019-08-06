using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class DefaultScoreCounter : IScoreCounter
{
    /*
     * Сейчас для каждой области циклом проходимся по всем спрайтам и считаем рейтинги,
     * а надо делать так чтобы за один проход цикла подсчитывались все рейтинги.
     */
    public virtual IEnumerable<(int, MyArea)> CountScores(IEnumerable<(int, MyArea)> result, MyColor[][][] sprites, IEnumerable<MyArea> allAreas, IAreaEnumerator areaEnumerator, ProgressReport progressReport, CancellationToken ct)
    {
        var scores = new ConcurrentDictionary<MyArea, int>();

        var areaList = allAreas.ToList();
        var areaCount = areaList.Count();

        Parallel.For(0, areaCount, (index, loopState) =>
        {
            if (ct.IsCancellationRequested)
                loopState.Stop();
            var area = areaList[index];
            var areaDimensions = area.Dimensions;

            var deletedOpaquePixels = 0;
            //areaEnumerator.EnumerateCopy(areaDimensions, (sprite, x, y) =>
            areaEnumerator.Enumerate(areaDimensions, (sprite, spriteIndex, x, y) =>
            {
                var comparedArea = MyArea.CreateFromSprite(sprite, x, y, areaDimensions);
                if (comparedArea.GetHashCode() == area.GetHashCode())
                {
                    //MyArea.EraseAreaFromSprite(sprite, x, y, areaDimensions);
                    deletedOpaquePixels += comparedArea.OpaquePixelsCount;
                }
            });

            var score = ((int)(Mathf.Pow(area.OpaquePixelsCount, 2f) / area.Dimensions.Square)) * deletedOpaquePixels;
            scores.AddOrUpdate(area, score, (key, _) => score);

            progressReport.OperationsDone++;
        });



        //var areaList = allAreas.ToList();
        //var areaCount = areaList.Count();

        //progressReport.OperationDescription = "Counting scores";
        //progressReport.OperationsCount = areaCount;
        //progressReport.OperationsDone = 0;

        //Parallel.For(0, areaCount, (index, loopState) =>
        //{
        //    if (ct.IsCancellationRequested)
        //        loopState.Stop();
        //    var candidate = areaList[index];
        //    var areaDimensions = candidate.Dimensions;

        //    var deletedOpaquePixels = 0;
        //    //areaEnumerator.EnumerateCopy(areaDimensions, (sprite, x, y) =>
        //    areaEnumerator.Enumerate(areaDimensions, (sprite, x, y) =>
        //    {
        //        var comparedArea = MyArea.CreateFromSprite(sprite, x, y, areaDimensions);
        //        if (comparedArea.GetHashCode() == candidate.GetHashCode())
        //        {
        //            //MyArea.EraseAreaFromSprite(sprite, x, y, areaDimensions);
        //            deletedOpaquePixels += comparedArea.OpaquePixelsCount;
        //        }
        //    });

        //    var score = ((int)(Mathf.Pow(candidate.OpaquePixelsCount, 2f) / candidate.Dimensions.Square)) * deletedOpaquePixels;
        //    scores.AddOrUpdate(candidate, score, (key, _) => score);

        //    progressReport.OperationsDone++;
        //});

        return scores.Select(kvp => (kvp.Value, kvp.Key));
    }
}
