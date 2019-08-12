using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;

public interface IAreaFetcher
{
    Task<ConcurrentDictionary<int, MyArea>> FetchAreas(MyColor[][][] sprites, IEnumerable<MyVector2> areaSizings, IAreaEnumerator areaEnumerator, ProgressReport progressReport);
}
