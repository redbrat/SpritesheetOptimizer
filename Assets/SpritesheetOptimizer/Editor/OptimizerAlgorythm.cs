using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class OptimizerAlgorythm
{
    public static bool Working;
    public static int CurrentOp;
    public static int CurrentOpsTotal;
    public static int ProcessedAreas;
    public static int UniqueAreas;
    public static int OpaquePixelsTotal;
    public static int UnoptimizedPixelsCount;
    public static long LastPassTime;

    private struct MyColor
    {
        private const short _byteMax = 256;
        private const int _2bytesMax = 65_536;
        private const long _4bytesMax = 4_294_967_296L;

        public readonly byte R;
        public readonly byte G;
        public readonly byte B;
        public readonly byte A;

        private readonly int _hash;

        public MyColor(byte r, byte g, byte b, byte a)
        {
            R = r;
            G = g;
            B = b;
            A = a;

            _hash = (R + G * _byteMax + B * _2bytesMax + A * _4bytesMax).GetHashCode();
        }

        public override int GetHashCode() => _hash;
    }

    private struct MyPoint
    {
        public readonly int Square;

        public readonly int X;
        public readonly int Y;

        public MyPoint(int x, int y)
        {
            X = x;
            Y = y;

            Square = x * y;
        }

        public override string ToString() => $"{X}, {Y}";
    }

    private struct MyRect
    {
        public readonly int X;
        public readonly int Y;
        public readonly int Width;
        public readonly int Height;

        public MyRect(int x, int y, int width, int height)
        {
            X = x;
            Y = y;
            Width = width;
            Height = height;
        }
    }

    private struct MyArea
    {
        public readonly MyColor[] _colors;

        public readonly MyPoint Dimensions;

        public readonly int OpaquePixelsCount;
        public readonly int PixelsCount;

        private readonly int _hash;

        public MyArea(MyPoint dimensions, params MyColor[] colors)
        {
            _colors = colors;
            Dimensions = dimensions;

            PixelsCount = _colors.Length;
            OpaquePixelsCount = 0;
            _hash = 0;
            for (int i = 0; i < _colors.Length; i++)
            {
                if (_colors[i].A > 0)
                    OpaquePixelsCount++;
                _hash += (i + 1) * _colors[i].GetHashCode() * short.MaxValue;
            }
            _hash *= dimensions.GetHashCode();
        }

        public override int GetHashCode() => _hash;

        public static bool ContainsOpaquePixels(MyColor[][] sprite, int x, int y, MyPoint dimensions)
        {
            for (int xx = 0; xx < dimensions.X; xx++)
                for (int yy = 0; yy < dimensions.Y; yy++)
                    if (sprite[x + xx][y + yy].A > 0)
                        return true;
            return false;
        }

        public static MyArea CreateFromSprite(MyColor[][] sprite, int x, int y, MyPoint dimensions)
        {
            var colors = new MyColor[dimensions.Square];
            for (int xx = 0; xx < dimensions.X; xx++)
                for (int yy = 0; yy < dimensions.Y; yy++)
                    colors[xx + yy * dimensions.X] = sprite[x + xx][y + yy];
            return new MyArea(dimensions, colors);
        }
        
        public static void EraseAreaFromSprite(MyColor[][] sprite, int x, int y, MyPoint dimensions)
        {
            for (int xx = 0; xx < dimensions.X; xx++)
                for (int yy = 0; yy < dimensions.Y; yy++)
                    sprite[x + xx][y + yy] = new MyColor(byte.MinValue, byte.MinValue, byte.MinValue, byte.MinValue);
        }

        public static void EraseUpdateEmptinessMap(MyColor[][] sprite, bool[][] spritesMapOfEmptiness, int x, int y, MyPoint erasedAreaDimensions, MyPoint updatedAreaDimensions)
        {
            for (int xx = 0; xx < erasedAreaDimensions.X; xx++)
                for (int yy = 0; yy < erasedAreaDimensions.Y; yy++)
                {
                    var spriteXCoord = x + xx;
                    var spriteYCoord = y + yy;
                    if (spriteXCoord + updatedAreaDimensions.X >= sprite.Length || spriteYCoord + updatedAreaDimensions.Y >= sprite[spriteXCoord].Length)
                        continue;
                    spritesMapOfEmptiness[spriteXCoord][spriteYCoord] = !ContainsOpaquePixels(sprite, spriteXCoord, spriteYCoord, updatedAreaDimensions);
                }
        }
    }

    private struct Chunk
    {
        public int Id;
        public Vector2Int Coordinates;
    }

    public static void Go(Vector2Int area, Sprite sprite)
    {
        Working = true;

        var texture = sprite.texture;
        var path = AssetDatabase.GetAssetPath(sprite);
        var ti = AssetImporter.GetAtPath(path) as TextureImporter;
        ti.isReadable = true;
        ti.SaveAndReimport();

        var sprites = new MyColor[ti.spritesheet.Length][][];
        for (int i = 0; i < ti.spritesheet.Length; i++)
        {
            var currentSprite = ti.spritesheet[i];

            var xOrigin = Mathf.FloorToInt(currentSprite.rect.x);
            var yOrigin = Mathf.CeilToInt(currentSprite.rect.y);
            var width = Mathf.CeilToInt(currentSprite.rect.width);
            var height = Mathf.CeilToInt(currentSprite.rect.height);
            var currentColors = new MyColor[width][];
            for (int x = 0; x < width; x++)
            {
                currentColors[x] = new MyColor[height];
                for (int y = 0; y < height; y++)
                {
                    var color = texture.GetPixel(xOrigin + x, yOrigin + y);
                    currentColors[x][y] = new MyColor(
                        Convert.ToByte(Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue)), 
                        Convert.ToByte(Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue)), 
                        Convert.ToByte(Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue)), 
                        Convert.ToByte(Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue))
                    );
                }
            }
            sprites[i] = currentColors;
        }

        Texture optimizedSpritesheet;

        //goGetEm(new MyPoint(area.x, area.y), sprites, out optimizedSpritesheet);
        var task = new Task<Chunk[][]>(() => goGetEm(new MyPoint(area.x, area.y), sprites/*, out optimizedSpritesheet*/));
        task.Start();
    }

    private const int _bestOfTheDirtyBufferSize = 8;

    private static Chunk[][] goGetEm(MyPoint area, MyColor[][][] sprites/*, out Texture optimizedSpritesheet*/)
    {
        //optimizedSpritesheet = new Texture2D(1, 1);
        //Debug.Log("goGetEm . ......"); 
        int pixelsTotal;
        int opaquePixelsTotal;

        countOpaquePixels(sprites, out pixelsTotal, out opaquePixelsTotal);

        OpaquePixelsTotal = pixelsTotal;
        UnoptimizedPixelsCount = opaquePixelsTotal;
        //Debug.Log($"pixelsTotal = {pixelsTotal}, opaquePixelsTotal = {opaquePixelsTotal}");

        var areaVariants = getAreaVariants(area);
        //Debug.Log("Area variants:");
        //for (int i = 0; i < areaVariants.Length; i++)
        //    Debug.Log($"    {areaVariants[i].ToString()}");

        //var areas = new Dictionary<int, MyArea>();
        //for (int i = 0; i < areaVariants.Length; i++)
        //{
        //    CurrentVariant++;
        //    CurrentSprite = 0;
        //    for (int j = 0; j < sprites.Length; j++)
        //    {
        //        CurrentSprite++;
        //        getUniqueAreas(areaVariants[i], sprites[j], areas);
        //    }
        //}

        var mapsOfEmptiness = new Dictionary<MyPoint, bool[][][]>();
        for (int i = 0; i < areaVariants.Length; i++)
        {
            var currentAreaVariant = areaVariants[i];
            var currentMapOfEmptiness = new bool[sprites.Length][][];
            for (int j = 0; j < sprites.Length; j++)
            {
                var sprite = sprites[j];
                var spriteMapOfEmptiness = new bool[sprite.Length - currentAreaVariant.X][];
                for (int x = 0; x < sprite.Length - currentAreaVariant.X; x++)
                {
                    spriteMapOfEmptiness[x] = new bool[sprite[x].Length - currentAreaVariant.Y];
                    for (int y = 0; y < sprite[x].Length - currentAreaVariant.Y; y++)
                        spriteMapOfEmptiness[x][y] = !MyArea.ContainsOpaquePixels(sprite, x, y, currentAreaVariant);
                }
                currentMapOfEmptiness[j] = spriteMapOfEmptiness;
            }
            mapsOfEmptiness.Add(currentAreaVariant, currentMapOfEmptiness);
        }

        var sw = new Stopwatch();

        while (UnoptimizedPixelsCount > 0)
        {
            sw.Reset();
            sw.Start();
            ProcessedAreas = 0;
            UniqueAreas = 0;

            var areas = new ConcurrentDictionary<int, MyArea>();
            var areaDirtyScores = new ConcurrentDictionary<int, long>(); //Dirty - потому что мы не удаляем пиксели по-настоящему, так что это рассчет грубый и неточный, но пойдет для первичного отсева.

            var overallOpsCount = areaVariants.Length * sprites.Length;

            CurrentOpsTotal = overallOpsCount;
            CurrentOp = 0;

            try
            {
                Parallel.For(0, overallOpsCount, (int index, ParallelLoopState state) =>
                {
                    if (state.IsExceptional)
                        Debug.Log("Exception!");
                    var i = Mathf.FloorToInt(index / sprites.Length);
                    var j = index - i * sprites.Length;
                    var targetArea = areaVariants[i];
                    var mapOfEmptinessForAreaAndSprite = mapsOfEmptiness[targetArea][j];
                    getUniqueAreas(targetArea, sprites[j], areas, areaDirtyScores, mapOfEmptinessForAreaAndSprite);
                });
            }
            catch (AggregateException ae)
            {
                Debug.Log("catch");
                ae.Handle((inner) =>
                {
                    Debug.Log(inner.Message);
                    return true;
                });
            }

            Debug.Log($"unique areas count = {areas.Count}");

            var orderedDirtyKvpArray = areaDirtyScores.ToArray().OrderByDescending(kvp => kvp.Value).ToArray();
            Debug.Log($"The winner is {orderedDirtyKvpArray[0].Key} with highest score of {orderedDirtyKvpArray[0].Value}!");
            for (int i = 0; i < _bestOfTheDirtyBufferSize; i++)
            {
                Debug.Log($"    {i + 1}. {orderedDirtyKvpArray[i].Key}. Score: {orderedDirtyKvpArray[i].Value}");
            }

            /*
             * Итак, что мы тут делаем. Мы имеем грязный словарь, с примерными значениями полезности. Что мы должны сделать в идеальном мире -
             * - пройтись с каждым из нескольких миллионов областей по потенциально многомиллионопиксельному спрайту и проанализировать каждый в плане
             * полезности. И сделать это надо для каждого шага. Ясно, что сложность тут зашкаливает. Поэтому, я думаю, надо иметь некий буфер, выборку
             * возможных кандитатов, например, 100. Берем 100 самых достойных областей (первые 100 из грязного словаря), проходимся с ними по спрайтам,
             * и смотрим какой из них удаляет больше всего пикселей. Его и забираем и действительно удаляем все пиксели с ним. У нас осталось на 1 область
             * грязного списка меньше. Повторять покуда остались пиксели.
             */


            //var imageCopy = CopyArrayOfColors(sprites);
            //var same = true;
            //for (int i = 0; i < imageCopy.Length; i++)
            //{
            //    for (int j = 0; j < imageCopy[i].Length; j++)
            //    {
            //        for (int m = 0; m < imageCopy[i][j].Length; m++)
            //        {
            //            if (imageCopy[i][j][m].GetHashCode() != sprites[i][j][m].GetHashCode())
            //                same = false;
            //        }
            //    }
            //}

            //if (same)
            //    Debug.Log($"Clone of image is same"); //Same!
            //else
            //    Debug.Log($"Clone of image isn't the same"); 

            CurrentOpsTotal = _bestOfTheDirtyBufferSize;
            CurrentOp = 0;

            var partialCleanScore = new ConcurrentDictionary<int, long>();

            Parallel.For(0, _bestOfTheDirtyBufferSize, index =>
            {
                var imageCopy = CopyArrayOf(sprites);
                var candidateHash = orderedDirtyKvpArray[index].Key;
                var candidate = areas[candidateHash];
                var areaDimensions = candidate.Dimensions;

                var emptinessMapCopy = CopyArrayOf(mapsOfEmptiness[areaDimensions]);

                var deletedOpaquePixels = 0;
                for (int i = 0; i < imageCopy.Length; i++)
                {
                    var sprite = imageCopy[i];
                    var spritesMapOfEmptiness = emptinessMapCopy[i];
                    for (int x = 0; x < sprite.Length - areaDimensions.X; x++)
                    {
                        for (int y = 0; y < sprite[x].Length - areaDimensions.Y; y++)
                        {
                            if (spritesMapOfEmptiness[x][y])
                                continue;
                            var comparedArea = MyArea.CreateFromSprite(sprite, x, y, areaDimensions);
                            if (comparedArea.GetHashCode() == candidate.GetHashCode())
                            {
                                MyArea.EraseAreaFromSprite(sprite, x, y, areaDimensions);
                                deletedOpaquePixels += comparedArea.OpaquePixelsCount;
                                MyArea.EraseUpdateEmptinessMap(sprite, spritesMapOfEmptiness, x, y, areaDimensions, areaDimensions); //т.к. мы сейчас смотрим только на эффективность текущей области в плане удаления пикселей, нам не нужно оптимизировать другие области
                            }
                        }
                    }
                }

                var cleanScore = ((long)(Mathf.Pow(candidate.OpaquePixelsCount, 2f) / candidate.Dimensions.Square)) * deletedOpaquePixels;
                partialCleanScore.AddOrUpdate(candidateHash, cleanScore, (key, _) => cleanScore);
                CurrentOp++;
            });

            var orderedCleanKvpArray = partialCleanScore.ToArray().OrderByDescending(kvp => kvp.Value).ToArray();

            Debug.Log($"");
            Debug.Log($"It's time for clean results everybody.");
            Debug.Log($"The winner is {orderedCleanKvpArray[0].Key} with highest score of {orderedCleanKvpArray[0].Value}!");
            for (int i = 0; i < orderedCleanKvpArray.Length; i++)
            {
                Debug.Log($"    {i + 1}. {orderedCleanKvpArray[i].Key}. Score: {orderedCleanKvpArray[i].Value}");
            }

            /*
             * Ок, теперь мы имеем чистый список и победителя - забираем его из areas, удаляем его пиксели с картинки, наносим на карту его id,
             * и после этого мы должны пойти по новой - пересчитать areaDirtyScores, взять buffer лучших, посчитать Clean, взять лучшего, и т.д..
             * Но вообще я могу сделать по-другому. Взять старый areaDirtyScores и пересчитать только те области, которые были затронуты предыдущим 
             * удалением. Для этого мне надо иметь карту размером с картинку, где каждый пиксель будет содержать инфу о том, частью какого хеша он является.
             * Поэтому при удалении пикселей победителя чистого списка с картинки, мы сохрянем все уникальные хеши удаленных пикселей, и потом пересчитываем
             * области с соответствующими хешами - может оказаться, что эти области вообще больше не существуют и надо их тогда удалить из грязного списка.
             * Если же они существуют - обновляем их рейтинг в грязном списке. А затем уже можно пойти по новой итерации цикла.
             * 
             * С другой стороны, карта размером с картинку - это потенциально много миллионов List'ов, каждый из которых будет содержать потенциально сотни 
             * значений. В ххудшем случае, если у нас 4к текстура и какая-нибудь большая в пределах разумного область, допустим, 8х8, то у нас 16 миллионов 
             * листов, и, несколько сотен хешей, размером, допустим 40 байт. В общем, не знаю, может я неправильно рассчитал, но у меня получилось, что мне 
             * понадобятся несколько сотен гигабайт оперативки для всего это счатья. Так что наверное, лучше все же смещать баланс в сторону вычислительной
             * сложности.
             * 
             * Нет, все-таки мне нужно как-то это дело оптимизировать. Так оставлять нельзя, очень долго будет обрабатываться. Я думаю, нужно сделать карту, 
             * содержащую информацию о пустых областях, чтобы можно было скипнуть проходы цикла.
             */

            var winnerAreaHash = orderedCleanKvpArray[0].Key;
            var winnerArea = areas[winnerAreaHash];
            var winnerAreaDimensions = winnerArea.Dimensions;
            var winnerAreaEmptinessMap = mapsOfEmptiness[winnerAreaDimensions];

            var opaquePixelsDeletedByWinner = 0;
            for (int i = 0; i < sprites.Length; i++)
            {
                var sprite = sprites[i];
                var spritesMapOfEmptiness = winnerAreaEmptinessMap[i];
                for (int x = 0; x < sprite.Length - winnerAreaDimensions.X; x++)
                {
                    for (int y = 0; y < sprite[x].Length - winnerAreaDimensions.Y; y++)
                    {
                        if (spritesMapOfEmptiness[x][y])
                            continue;
                        var comparedArea = MyArea.CreateFromSprite(sprite, x, y, winnerAreaDimensions);
                        if (comparedArea.GetHashCode() == winnerArea.GetHashCode())
                        {
                            MyArea.EraseAreaFromSprite(sprite, x, y, winnerAreaDimensions);
                            opaquePixelsDeletedByWinner += comparedArea.OpaquePixelsCount;

                            /*
                             * Сообщаем все мапам пустот, что в данной конкретной области на данном конкретном спрайте прибавилось пустоты, поэтому их надо 
                             * обновить. Это действует на все варианты областей и только на 1 конкретный спрайт.
                             */

                            for (int v = 0; v < areaVariants.Length; v++)
                            {
                                var currentAreaVariant = areaVariants[v];
                                MyArea.EraseUpdateEmptinessMap(sprite, spritesMapOfEmptiness, x, y, winnerAreaDimensions, currentAreaVariant);
                            }
                        }
                    }
                }
            }

            UnoptimizedPixelsCount -= opaquePixelsDeletedByWinner;
            sw.Stop();
            LastPassTime = sw.ElapsedMilliseconds;
        }
        
        Working = false; 

        throw new NotImplementedException(); 
        //return new Chunk[0][]; 
    }

    private static T[][][] CopyArrayOf<T>(T[][][] source)
    {
        var len = source.Length;
        var dest = new T[len][][];

        for (int i = 0; i < len; i++)
        {
            var len2 = source[i].Length;
            dest[i] = new T[len2][];

            for (int j = 0; j < len2; j++)
            {
                var len3 = source[i][j].Length;
                dest[i][j] = new T[len3];

                Array.Copy(source[i][j], dest[i][j], len3);
            }
        }

        return dest;
    }

    private static void getUniqueAreas(MyPoint areaResolution, MyColor[][] sprite, ConcurrentDictionary<int, MyArea> areas, ConcurrentDictionary<int, long> areaDirtyScores, bool[][] mapOfEmptiness)
    {
        var areaSquare = areaResolution.X * areaResolution.Y;
        for (int x = 0; x < sprite.Length - areaResolution.X; x++)
        {
            for (int y = 0; y < sprite[x].Length - areaResolution.Y; y++)
            {
                ProcessedAreas++;
                if (mapOfEmptiness[x][y])
                    continue;
                var area = MyArea.CreateFromSprite(sprite, x, y, areaResolution);
                var hash = area.GetHashCode();
                if (areas.TryAdd(hash, area))
                    UniqueAreas++;

                var dirtyScore = (int)(Mathf.Pow(area.OpaquePixelsCount, 3f) / areaSquare);
                areaDirtyScores.AddOrUpdate(hash, dirtyScore, (key, existingValue) => existingValue + dirtyScore);
            }
        }
        CurrentOp++;
    }

    private static MyPoint[] getAreaVariants(MyPoint area)
    {
        var resultList = new List<MyPoint>();

        var currentArea = area;
        var counter = area.X != area.Y ? area.X < area.Y ? 1 : 2 : 0;
        while (currentArea.X > 0 && currentArea.Y > 0)
        {
            resultList.Add(currentArea);
            if (counter % 3 == 0)
                currentArea = new MyPoint(currentArea.X - 1, currentArea.Y);
            else if (counter % 3 == 1)
                currentArea = new MyPoint(currentArea.X + 1, currentArea.Y - 1);
            else
                currentArea = new MyPoint(currentArea.X - 1, currentArea.Y);

            counter++;
        }

        return resultList.ToArray();
    }

    private static void countOpaquePixels(MyColor[][][] sprites, out int pixelsTotal, out int opaquePixelsTotal)
    {
        pixelsTotal = 0;
        opaquePixelsTotal = 0;

        for (int i = 0; i < sprites.Length; i++)
        {
            for (int x = 0; x < sprites[i].Length; x++)
            {
                for (int y = 0; y < sprites[i][x].Length; y++)
                {
                    pixelsTotal++;
                    if (sprites[i][x][y].A > 0)
                        opaquePixelsTotal++;
                }
            }
        }
    }
}
