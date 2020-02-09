using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

[CreateAssetMenu(fileName = nameof(GetCudaExport), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.FunctionsName + "/" + nameof(GetCudaExport), order = 0)]
public class GetCudaExport : FunctionBase1<SpritesInfo, byte[]>
{
    public override byte[] Invoke(SpritesInfo param)
    {
        var colorsResults = param.Colors;
        var sizings = 8.GetStandardSizingsOfASquare(); //Пока всегда будем плясать стандартно от 8
        var sizingsDeconstructed = sizings.Select(s => new short[] { (short)s.X, (short)s.Y }).ToArray();
        var sizingsCount = (short)sizingsDeconstructed.Length;

        var spritesCount = (short)param.Bytes.Length;

        var metaList = new List<FormatMetaSpriteInfo>();
        for (int i = 0; i < spritesCount; i++)
        {
            var currentSprite = param.Sprites[i];
            var pathToSprite = AssetDatabase.GetAssetPath(currentSprite);
            metaList.Add(new FormatMetaSpriteInfo(pathToSprite, currentSprite.name));
        }
        var meta = new FormatMeta(metaList.ToArray());


        var parallelizedSizingsList = new List<byte>();
        for (int i = 0; i < sizingsDeconstructed.Length; i++)
            parallelizedSizingsList.AddRange(BitConverter.GetBytes(sizingsDeconstructed[i][0]));
        for (int i = 0; i < sizingsDeconstructed.Length; i++)
            parallelizedSizingsList.AddRange(BitConverter.GetBytes(sizingsDeconstructed[i][1]));


        var dataList = new List<byte>();
        var registry = new List<RegistryStruct>();

        var byteLineLength = 0;
        for (int i = 0; i < spritesCount; i++)
        {
            var currentSpriteBytes = param.Bytes[i];
            var width = currentSpriteBytes.Length;
            var height = currentSpriteBytes[0].Length;

            var widthAndHeight = width << 16 | height;
            var registryEntry = new RegistryStruct();
            registryEntry.SpritesByteOffset = dataList.Count;
            registryEntry.WidthAndHeight = widthAndHeight;
            registry.Add(registryEntry);

            byteLineLength += width * height;

            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                {
                    var val = currentSpriteBytes[x][y][0];
                    dataList.Add(val);
                }
        }

        for (int i = 0; i < spritesCount; i++)
        {
            var currentSpriteBytes = param.Bytes[i];
            var width = currentSpriteBytes.Length;
            var height = currentSpriteBytes[0].Length;

            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                {
                    var val = currentSpriteBytes[x][y][1];
                    dataList.Add(val);
                }
        }

        for (int i = 0; i < spritesCount; i++)
        {
            var currentSpriteBytes = param.Bytes[i];
            var width = currentSpriteBytes.Length;
            var height = currentSpriteBytes[0].Length;

            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                {
                    var val = currentSpriteBytes[x][y][2];
                    dataList.Add(val);
                }
        }

        for (int i = 0; i < spritesCount; i++)
        {
            var currentSpriteBytes = param.Bytes[i];
            var width = currentSpriteBytes.Length;
            var height = currentSpriteBytes[0].Length;

            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                {
                    var val = currentSpriteBytes[x][y][3];
                    dataList.Add(val);
                }
        }

        /*
         * Войдмапы должны храниться по спрайтам, а там по сайзингам, повторяя структуру куды для лучшего затем прасинга.
         * 
         * Надо будет потом проверить правильность войд-мап...
         */

        var voidMaps = new List<List<List<byte>>>();
        var voidMapsOffsets = new List<List<int>>();
        var voidMapsLength = 0;
        var currentVoidOffset = 0;
        for (int i = 0; i < spritesCount; i++)
        {
            var newSpriteVoidMaps = new List<List<byte>>();
            var newSpriteVoidMapsOffsets = new List<int>();

            var currentSpriteBytes = param.Bytes[i];
            var width = currentSpriteBytes.Length;
            var height = currentSpriteBytes[0].Length;
            for (int j = 0; j < sizingsDeconstructed.Length; j++)
            {
                newSpriteVoidMapsOffsets.Add(currentVoidOffset);
                var newSpriteSizingVoidMap = new List<byte>();
                var bitsCounter = (long)0;

                var sizing = sizingsDeconstructed[j];
                var sizingWidth = sizing[0];
                var sizingHeight = sizing[1];
                for (int x = 0; x < width - sizingWidth + 1; x++)
                {
                    for (int y = 0; y < height - sizingHeight + 1; y++)
                    {
                        var isVoid = true;
                        for (int xx = 0; xx < sizingWidth; xx++)
                        {
                            for (int yy = 0; yy < sizingHeight; yy++)
                            {
                                var alpha = currentSpriteBytes[x + xx][y + yy][3];
                                if (alpha != 0)
                                {
                                    isVoid = false;
                                    break;
                                }
                            }

                            if (!isVoid)
                                break;
                        }
                        writeIntoBitsList(newSpriteSizingVoidMap, bitsCounter++, isVoid ? 0 : 1);
                    }
                }

                currentVoidOffset += newSpriteSizingVoidMap.Count;
                voidMapsLength += newSpriteSizingVoidMap.Count;
                newSpriteVoidMaps.Add(newSpriteSizingVoidMap);
            }

            voidMaps.Add(newSpriteVoidMaps);
            voidMapsOffsets.Add(newSpriteVoidMapsOffsets);
        }

        var rFlags = new List<List<byte>>();
        var gFlags = new List<List<byte>>();
        var bFlags = new List<List<byte>>();
        var aFlags = new List<List<byte>>();
        var flagsLineLength = 0;
        var opaquePixelsCount = (uint)0;
        for (int i = 0; i < spritesCount; i++)
        {
            var newRFlagsList = new List<byte>();
            var newGFlagsList = new List<byte>();
            var newBFlagsList = new List<byte>();
            var newAFlagsList = new List<byte>();
            var bitsCounter = (long)0;

            var currentSpriteBytes = param.Bytes[i];
            var width = currentSpriteBytes.Length;
            var height = currentSpriteBytes[0].Length;
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    var r = currentSpriteBytes[x][y][0];
                    var g = currentSpriteBytes[x][y][1];
                    var b = currentSpriteBytes[x][y][2];
                    var a = currentSpriteBytes[x][y][3];
                    writeIntoBitsList(newRFlagsList, bitsCounter, r > 127 ? 1 : 0);
                    writeIntoBitsList(newGFlagsList, bitsCounter, g > 127 ? 1 : 0);
                    writeIntoBitsList(newBFlagsList, bitsCounter, b > 127 ? 1 : 0);
                    writeIntoBitsList(newAFlagsList, bitsCounter, a > 127 ? 1 : 0);

                    if (a != 0)
                        opaquePixelsCount++;
                    bitsCounter++;
                }
            }

            rFlags.Add(newRFlagsList);
            gFlags.Add(newGFlagsList);
            bFlags.Add(newBFlagsList);
            aFlags.Add(newAFlagsList);
            var registryEntry = registry[i];
            registry[i] = registryEntry;
            flagsLineLength += newRFlagsList.Count;
        }

        Debug.Log($"opaquePixelsCount = {opaquePixelsCount}");

        var registryParalellized = new List<byte>();
        for (int i = 0; i < registry.Count; i++)
            registryParalellized.AddRange(BitConverter.GetBytes(registry[i].SpritesByteOffset));
        for (int i = 0; i < registry.Count; i++)
            registryParalellized.AddRange(BitConverter.GetBytes((short)(registry[i].WidthAndHeight >> 16 & 65535)));
        for (int i = 0; i < registry.Count; i++)
            registryParalellized.AddRange(BitConverter.GetBytes((short)(registry[i].WidthAndHeight & 65535)));



        var combinedData = new byte[
            2 //Зарезервировано
            + sizeof(short) //spritesCount
            + sizeof(short) //sizingsCount
            + sizingsCount * sizeof(short) * 2 //структура сайзинга - это два шорта (х и у)
            + spritesCount * (sizeof(int) * 2 + sizeof(short) * 2) //текущая структура регистра состоит из 2 интов и 2 шортов на спрайт
            + sizeof(int) //byteLineLength - длина канала данных в байтах
            + byteLineLength * 4 //Дальше идут собственно данные - 4 канала (r, g, b, a)

            //Дальше вспомогательные

            + sizingsCount * spritesCount * sizeof(int) //Это регистр оффсетов пустот. Там оффсеты на каждую карту пустот для каждого спрайта и каждого сайзинга.
            + sizeof(int) // voidMapsLength - длина всей карты пустот в байтах (на самом деле она в битах), но тут именно длина блоба
            + voidMapsLength //Собсно она, в байтах
            + sizeof(int) // flagsLineLength - длина 1 канала флагов для данных в байтах (на самом деле она в битах), но тут именно длина блоба
            + flagsLineLength * 4 // 4 канала флагов - по одному для каждого канала данных
            + sizeof(uint) //ScoresCount - кол-во возможных областей всего
            + spritesCount * sizingsCount * sizeof(uint) //байтовые сдвиги рабочих областей спрайтов. Оказывается у меня их пока не было. Сдвиги войдмап не в счет, т.к. там /8
            + sizeof(uint) //OpaquePixelsCount
            ];

        combinedData[0] = 0; //Эти два байта 
        combinedData[1] = 0; //зарезервированы
        combinedData[2] = (byte)(spritesCount & 255);
        combinedData[3] = (byte)(spritesCount >> 8 & 255);

        combinedData[4] = (byte)(sizingsCount & 255);
        combinedData[5] = (byte)(sizingsCount >> 8 & 255);

        var currentOffset = 6;
        for (int i = 0; i < parallelizedSizingsList.Count; i++)
            combinedData[currentOffset + i] = parallelizedSizingsList[i];
        currentOffset += parallelizedSizingsList.Count;

        for (int i = 0; i < registryParalellized.Count; i++)
            combinedData[currentOffset + i] = registryParalellized[i];
        currentOffset += registryParalellized.Count;

        combinedData[currentOffset++] = (byte)(byteLineLength & 255);
        combinedData[currentOffset++] = (byte)(byteLineLength >> 8 & 255);
        combinedData[currentOffset++] = (byte)(byteLineLength >> 16 & 255);
        combinedData[currentOffset++] = (byte)(byteLineLength >> 24 & 255);

        for (int i = 0; i < dataList.Count; i++)
            combinedData[currentOffset + i] = dataList[i];
        currentOffset += dataList.Count;


        //Вспомогательные данные.

        for (int i = 0; i < voidMapsOffsets.Count; i++)
        {
            for (int j = 0; j < voidMapsOffsets[i].Count; j++)
            {
                combinedData[currentOffset++] = (byte)(voidMapsOffsets[i][j] & 255);
                combinedData[currentOffset++] = (byte)(voidMapsOffsets[i][j] >> 8 & 255);
                combinedData[currentOffset++] = (byte)(voidMapsOffsets[i][j] >> 16 & 255);
                combinedData[currentOffset++] = (byte)(voidMapsOffsets[i][j] >> 24 & 255);
            }
        }

        combinedData[currentOffset++] = (byte)(voidMapsLength & 255);
        combinedData[currentOffset++] = (byte)(voidMapsLength >> 8 & 255);
        combinedData[currentOffset++] = (byte)(voidMapsLength >> 16 & 255);
        combinedData[currentOffset++] = (byte)(voidMapsLength >> 24 & 255);

        for (int i = 0; i < voidMaps.Count; i++)
        {
            for (int j = 0; j < voidMaps[i].Count; j++)
            {
                for (int m = 0; m < voidMaps[i][j].Count; m++)
                {
                    combinedData[currentOffset++] = voidMaps[i][j][m];
                }
            }
        }

        Debug.Log($"flagsCount = {flagsLineLength}");
        combinedData[currentOffset++] = (byte)(flagsLineLength & 255);
        combinedData[currentOffset++] = (byte)(flagsLineLength >> 8 & 255);
        combinedData[currentOffset++] = (byte)(flagsLineLength >> 16 & 255);
        combinedData[currentOffset++] = (byte)(flagsLineLength >> 24 & 255);

        for (int i = 0; i < rFlags.Count; i++)
            for (int j = 0; j < rFlags[i].Count; j++)
                combinedData[currentOffset++] = rFlags[i][j];

        for (int i = 0; i < gFlags.Count; i++)
            for (int j = 0; j < gFlags[i].Count; j++)
                combinedData[currentOffset++] = gFlags[i][j];

        for (int i = 0; i < bFlags.Count; i++)
            for (int j = 0; j < bFlags[i].Count; j++)
                combinedData[currentOffset++] = bFlags[i][j];

        for (int i = 0; i < aFlags.Count; i++)
            for (int j = 0; j < aFlags[i].Count; j++)
                combinedData[currentOffset++] = aFlags[i][j];


        var scoresCount = (uint)0;
        var workingSpriteOffsets = new uint[spritesCount * sizingsCount];
        for (int i = 0; i < spritesCount; i++)
        {
            var currentSpriteBytes = param.Colors[i];
            var spriteWidth = currentSpriteBytes.Length;
            var spriteHeight = currentSpriteBytes[0].Length;

            for (int j = 0; j < sizingsCount; j++)
            {
                var sizing = sizingsDeconstructed[j];
                var sizingWidth = sizing[0];
                var sizingHeight = sizing[1];

                var currentWorkingSpriteLength = (uint)((spriteWidth - sizingWidth + 1) * (spriteHeight - sizingHeight + 1));
                workingSpriteOffsets[i * sizingsCount + j] = scoresCount;
                scoresCount += currentWorkingSpriteLength;
            }
        }

        Debug.Log($"scoresCount = {scoresCount}");

        combinedData[currentOffset++] = (byte)(scoresCount & 255);
        combinedData[currentOffset++] = (byte)(scoresCount >> 8 & 255);
        combinedData[currentOffset++] = (byte)(scoresCount >> 16 & 255);
        combinedData[currentOffset++] = (byte)(scoresCount >> 24 & 255);

        for (int i = 0; i < workingSpriteOffsets.Length; i++)
        {
            combinedData[currentOffset++] = (byte)(workingSpriteOffsets[i] & 255);
            combinedData[currentOffset++] = (byte)(workingSpriteOffsets[i] >> 8 & 255);
            combinedData[currentOffset++] = (byte)(workingSpriteOffsets[i] >> 16 & 255);
            combinedData[currentOffset++] = (byte)(workingSpriteOffsets[i] >> 24 & 255);
        }


        combinedData[currentOffset++] = (byte)(opaquePixelsCount & 255);
        combinedData[currentOffset++] = (byte)(opaquePixelsCount >> 8 & 255);
        combinedData[currentOffset++] = (byte)(opaquePixelsCount >> 16 & 255);
        combinedData[currentOffset++] = (byte)(opaquePixelsCount >> 24 & 255);


        var metaTextBytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(meta));
        var metaLength = metaTextBytes.Length;
        var header = new byte[metaLength + 4];
        header[0] = (byte)(metaLength & 255);
        header[1] = (byte)(metaLength >> 8 & 255);
        header[2] = (byte)(metaLength >> 16 & 255);
        header[3] = (byte)(metaLength >> 24 & 255);
        Buffer.BlockCopy(metaTextBytes, 0, header, 4, metaLength);

        var finalBlob = new byte[header.Length + combinedData.Length];
        Buffer.BlockCopy(header, 0, finalBlob, 0, header.Length);
        Buffer.BlockCopy(combinedData, 0, finalBlob, header.Length, combinedData.Length);

        return finalBlob;
    }

    private void writeIntoBitsList(List<byte> bitsContainer, long bitsCounter, int bit)
    {
        var reminder = (int)(bitsCounter % 8);
        if (reminder == 0)
            bitsContainer.Add(0);
        bitsContainer[bitsContainer.Count - 1] |= (byte)((bit & 1) << reminder);
    }
}
