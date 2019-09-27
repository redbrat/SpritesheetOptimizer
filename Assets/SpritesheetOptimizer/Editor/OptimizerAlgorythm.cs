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
    public static int UnoptimizedPixelsCount;
    public static long LastPassTime;

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
        var task = new Task<Chunk[][]>(() => goGetEm(new MyVector2(area.x, area.y), sprites/*, out optimizedSpritesheet*/));
        task.Start();
    }

    private const int _bestOfTheDirtyBufferSize = 8;

    private static Chunk[][] goGetEm(MyVector2 area, MyColor[][][] sprites/*, out Texture optimizedSpritesheet*/)
    {
        //optimizedSpritesheet = new Texture2D(1, 1);
        //Debug.Log("goGetEm . ......"); 
        int pixelsTotal;
        int opaquePixelsTotal;

        countOpaquePixels(sprites, out pixelsTotal, out opaquePixelsTotal);

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

        var mapsOfEmptiness = new Dictionary<MyVector2, bool[][][]>();
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

        var map = new Dictionary<MyArea, List<(int, int, int)>>();

        var sw = new Stopwatch();
        while (UnoptimizedPixelsCount > 0)
        {
            sw.Reset();
            sw.Start();
            ProcessedAreas = 0;
            UniqueAreas = 0;

            var areas = new ConcurrentDictionary<int, MyArea>(); //Тут мы храним все уникальные области по их хешам
            var areaDirtyScores = new ConcurrentDictionary<int, long>(); //А тут - их счет по их хешам
            //Dirty - потому что мы не удаляем пиксели по-настоящему, так что счет может быть выше чем на самом деле из-за повторов.
            //Так что это рассчет грубый и неточный, но пойдет для первичного отсева.

            //Сейчас мы идем заполнять эти два словаря, и это можно делать параллельно - слава TPL
            var overallOpsCount = areaVariants.Length * sprites.Length;

            CurrentOpsTotal = overallOpsCount;
            CurrentOp = 0;

            try
            {
                Parallel.For(0, overallOpsCount, (int index, ParallelLoopState state) =>
                {
                    if (state.IsExceptional)
                        Debug.Log("Exception!");
                    var areaVariantIndex = Mathf.FloorToInt(index / sprites.Length);
                    var spriteIndex = index - areaVariantIndex * sprites.Length;
                    var targetArea = areaVariants[areaVariantIndex];
                    var mapOfEmptinessForAreaAndSprite = mapsOfEmptiness[targetArea][spriteIndex];
                    getUniqueAreas(targetArea, sprites[spriteIndex], spriteIndex, areas, areaDirtyScores, mapOfEmptinessForAreaAndSprite);
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

            /*
             * Итак, что мы тут делаем. Мы имеем грязный словарь, с примерными значениями полезности. Что мы должны сделать в идеальном мире -
             * - пройтись с каждым из нескольких миллионов областей по потенциально многомиллионопиксельному спрайту и проанализировать каждый в плане
             * полезности. И сделать это надо для каждого шага. Ясно, что сложность тут нереальная. Поэтому, я думаю, надо иметь некий буфер, выборку
             * возможных кандитатов, например, 100. Берем 100 самых достойных областей (первые 100 из грязного словаря), проходимся с ними по спрайтам,
             * и смотрим какой из них удаляет больше всего пикселей. Его и забираем и действительно удаляем все пиксели с ним. У нас осталось на 1 область
             * грязного списка меньше. Повторять покуда остались пиксели.
             */

            var orderedDirtyKvpArray = areaDirtyScores.ToArray().OrderByDescending(kvp => kvp.Value).ToArray();
            Debug.Log($"The winner is {orderedDirtyKvpArray[0].Key} with highest score of {orderedDirtyKvpArray[0].Value}!");
            for (int i = 0; i < _bestOfTheDirtyBufferSize; i++)
            {
                Debug.Log($"    {i + 1}. {orderedDirtyKvpArray[i].Key}. Score: {orderedDirtyKvpArray[i].Value}");
            }


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

            /*
             * Ок, дальше мы идем делать чистую проверку самых вероятных победителей - с пробным удалением пикселей.
             */

            CurrentOpsTotal = _bestOfTheDirtyBufferSize;
            CurrentOp = 0;

            var partialCleanScore = new ConcurrentDictionary<int, long>(); //Это и есть словарь с чистым счетом - при пробном удалении пикселей

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
                    var spriteCopy = imageCopy[i];
                    var spritesMapOfEmptinessCopy = emptinessMapCopy[i];
                    for (int x = 0; x < spriteCopy.Length - areaDimensions.X; x++)
                    {
                        for (int y = 0; y < spriteCopy[x].Length - areaDimensions.Y; y++)
                        {
                            if (spritesMapOfEmptinessCopy[x][y])
                                continue;
                            var comparedArea = MyArea.CreateFromSprite(spriteCopy, i, x, y, areaDimensions);
                            if (comparedArea.GetHashCode() == candidate.GetHashCode())
                            {
                                MyArea.EraseAreaFromSprite(spriteCopy, x, y, areaDimensions);
                                deletedOpaquePixels += comparedArea.OpaquePixelsCount;
                                MyArea.EraseUpdateEmptinessMap(spriteCopy, spritesMapOfEmptinessCopy, x, y, areaDimensions, areaDimensions); //т.к. мы сейчас смотрим только на эффективность текущей области в плане удаления пикселей, нам не нужно оптимизировать другие области
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
            Debug.Log($"It's time for clean results everybody!");
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

            var mappedAreas = new List<(int, int, int)>();

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
                        var comparedArea = MyArea.CreateFromSprite(sprite, i, x, y, winnerAreaDimensions);
                        if (comparedArea.GetHashCode() == winnerArea.GetHashCode())
                        {
                            MyArea.EraseAreaFromSprite(sprite, x, y, winnerAreaDimensions);
                            opaquePixelsDeletedByWinner += comparedArea.OpaquePixelsCount;

                            mappedAreas.Add((i, x, y));

                            /*
                             * Сообщаем всем мапам пустот, что в данной конкретной области на данном конкретном спрайте прибавилось пустоты, поэтому их 
                             * надо обновить. Это действует на все варианты областей и только на 1 конкретный спрайт.
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

            map.Add(winnerArea, mappedAreas);

            UnoptimizedPixelsCount -= opaquePixelsDeletedByWinner;
            sw.Stop();
            LastPassTime = sw.ElapsedMilliseconds;
        }

        /*
         * Ок, все равно занимает кучу времени. И еще почему-то кол-во уникальных областей неуклонно растет, хотя по логике, удаляя пиксели, мы должны
         * получать меньше уникальных областей. Хотя нет. Удаляя пиксели мы получаем много вырезанных пространств, много пробелов в картинке, и эти 
         * пробелы по идее должны добавлять уникальных вариантов составов пикселей.
         * 
         * Ок, так как мне уменьшить время выполнения? Я думаю, надо добавлять эвристик. Например, мы можем предположить, что скорее всего удаление 
         * какой-либо области из карты после одного прохода цикла не изменило ситуацию настолько кардинально, чтобы перелопатить все сверху донизу. 
         * Т.е. скорее всего все изменилось не настолько чтобы, например, каждый из первых 10_000 областей из лидеров грязного списка перестал быть в 
         * первых 10_000 после удаления. На самом деле, удаление потенциально затрагивает не так уж много областей. В области 4х4 может быть 16 1х1 
         * уникальных областей, 9 2х2 уникальных областей, и 4 3х3 уникальных областей, т.е. всего, кроме собственно самой 4х4 области победителя, могут
         * быть потенциально затронуты лишь 29 областей. Нет, это те, что могут быть полностью удалены, а затронуты -  15+16*(кол-во вариантов областей - 1), 
         * то есть, для 4х4 - это ((16 * 10) - 1) областей, т.е. 159 затронутых областей. Нет, забыл учесть те, что слева и сверху, их должно быть еще порядка
         * 16 * 3 * 10. Т.е., если считать приблизительно, для 4х4 и 10 вариантов областей кол-во затронутых областей не должно быть больше 4х4х4х10. Скорее 
         * всего их будет где-то в районе 3х4х4х10 * 0.5 + 4х4х10, или 2.5х4х4х10 или 400. Т.е. в принципе, если предположить худший сценарий, что все эти
         * 400 областей были в 400 первых в грязном списке, а после удаления они все ушли из этих 400 первых, то нам надо пересчитать рейтинги только для 
         * первых 400 областей и переупорядочить список. Конечно, мы скорее всего не пересчитаем все реально затронутые области, но если их нету в первых 
         * 400 областях, значит эти затронутые области были довольно незначитальными по своему вкладу в картинку и не стали значительнее после удаления.
         * Т.е. да, они останутся с неправильным рейтингом за пределами 400 первых, но если вдруг на каком-то этапе они попадут в эти 400 первых мы 
         * пересчитаем их рейтинг. Т.е. по сути мы как бы говорим "да, после удаления рейтинг многих областей поменялся, но скорее всего после удаления 
         * новый чистый лидер будет среди первых (400 + buffer) грязных областей по версии до удаления".
         * 
         * Но, тут еще есть один момент - новые пустоты могут насоздавать новых областей, которые в принципе могут посоперничать с областями из лидеров. Т.е.
         * в принципе после удаления мы можем дополнительно обновлять список уникальных областей теми, что были обнаружены в прилежащих к удаленным областям,
         * считать для них рейтинги и потом включить их в общий список после пересчета первых 400 областей и перед переупорядочиванием.
         * 
         * Так, отмена, пиксели за пределами удаленных областей могут быть любыми, так что тут не 2.5х4х4х10 а гораздо больше областей может быть затронуто.
         * Тогда можно такую эвристику придумать. Если мы удалили процент A картинки, то затронутых пикселей областей потенциально будет процент B (где-то 
         * в районе Aх9). Т.е. в худшем случае верхний процент B областей грязного рейтинга ушел куда-то на дно. Т.е. достаточно перепроверить эти B 
         * процентов областей сверху грязного рейтинга и тогда мы сможем безопасно взять оттуда буффер. Конечно, это не меняет тот факт, что надо добавить 
         * новообразованных областей из окресностей удаленных.
         * 
         * Чтобы понять где окресности удаленных областей, мне нужна карта. Карта будет словарем с ключами в виде областей и значениями в виде координат 
         * вида (индекс спрайта, X, Y). Имея этот словарь и зная последнюю добавленную туда область мы сможем узнать все затронутые ей части картинки и 
         * пройтись по всем окресностям, создавая новые области по необходимости. Также эта карта будет полезна вдальнейшем при составлении структур 
         * областей для каждого спрайта.
         */

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

    private static void getUniqueAreas(MyVector2 areaResolution, MyColor[][] sprite, int spriteIndex, ConcurrentDictionary<int, MyArea> areas, ConcurrentDictionary<int, long> areaDirtyScores, bool[][] mapOfEmptiness)
    {
        var areaSquare = areaResolution.X * areaResolution.Y;
        for (int x = 0; x < sprite.Length - areaResolution.X; x++)
        {
            for (int y = 0; y < sprite[x].Length - areaResolution.Y; y++)
            {
                ProcessedAreas++;
                if (mapOfEmptiness[x][y])
                    continue;
                var area = MyArea.CreateFromSprite(sprite, spriteIndex, x, y, areaResolution);
                var hash = area.GetHashCode();
                if (areas.TryAdd(hash, area))
                    UniqueAreas++;

                var dirtyScore = (int)(Mathf.Pow(area.OpaquePixelsCount, 3f) / areaSquare);
                areaDirtyScores.AddOrUpdate(hash, dirtyScore, (key, existingValue) => existingValue + dirtyScore);
            }
        }
        CurrentOp++;
    }

    private static MyVector2[] getAreaVariants(MyVector2 area)
    {
        var resultList = new List<MyVector2>();

        var currentArea = area;
        var counter = area.X != area.Y ? area.X < area.Y ? 1 : 2 : 0;
        while (currentArea.X > 0 && currentArea.Y > 0)
        {
            resultList.Add(currentArea);
            if (counter % 3 == 0)
                currentArea = new MyVector2(currentArea.X - 1, currentArea.Y);
            else if (counter % 3 == 1)
                currentArea = new MyVector2(currentArea.X + 1, currentArea.Y - 1);
            else
                currentArea = new MyVector2(currentArea.X - 1, currentArea.Y);

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
