using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

public class MapOfEmptiness
{
    private const int _yMultiplier = 10_000;
    private readonly ConcurrentDictionary<MyVector2, ConcurrentDictionary<int, List<int>>> _mapOfEmptiness;

    public MapOfEmptiness()
    {
        _mapOfEmptiness = new ConcurrentDictionary<MyVector2, ConcurrentDictionary<int, List<int>>>();
    }

    internal async Task Initialize(IEnumerable<MyVector2> areaSizings, MyColor[][][] sprites, IAreaEnumerator enumerator)
    {
        var sizingsList = areaSizings.ToList();
        await Task.Run(() =>
        {
            Parallel.For(0, sizingsList.Count, i =>
            {
                var sizing = sizingsList[i];
                var newSizingDict = new ConcurrentDictionary<int, List<int>>();
                _mapOfEmptiness.AddOrUpdate(sizing, newSizingDict, (_, __) => newSizingDict);
                Parallel.For(0, sprites.Length, j =>
                {
                    var newSpriteDict = new List<int>();
                    newSizingDict.AddOrUpdate(j, newSpriteDict, (_, __) => newSpriteDict);
                    enumerator.EnumerateThroughSprite(sizing, j, (sprite, spriteIndex, x, y) =>
                    {
                        var area = MyArea.CreateFromSprite(sprite, spriteIndex, x, y, sizing);
                        if (area.OpaquePixelsCount == 0)
                            newSpriteDict.Add(x + y * _yMultiplier);
                    });
                });
            });
        });
    }

    internal bool Contains(MyVector2 areaSizing, int spriteIndex, int x, int y) => _mapOfEmptiness[areaSizing][spriteIndex].Contains(x + y * _yMultiplier);

    internal void MakeEmpty(MyVector2 emptyArea, int spriteIndex, int spriteCoordX, int spriteCoordY)
    {
        foreach (var kvp in _mapOfEmptiness)
        {
            var area = kvp.Key;
            if (!area.CanBePutInto(emptyArea))
                continue;
            var spritesDict = kvp.Value;
            if (!spritesDict.ContainsKey(spriteIndex))
                continue;
            var emptinessList = spritesDict[spriteIndex];
            var width = emptyArea.X - area.X;
            var height = emptyArea.Y - area.Y;
            for (int x = 0; x <= width; x++)
            {
                for (int y = 0; y <= height; y++)
                {
                    var coords = (spriteCoordX + x) + (spriteCoordY + y) * _yMultiplier;
                    if (!emptinessList.Contains(coords))
                        emptinessList.Add(coords);
                }
            }
        }
    }
}
