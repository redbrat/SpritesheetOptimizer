using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine;

public class GpuAreaFetcher : IAreaFetcher
{
    public ConcurrentDictionary<int, MyArea> FetchAreas(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport)
    {
        throw new System.NotImplementedException();
    }
}
