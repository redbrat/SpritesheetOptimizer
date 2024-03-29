﻿using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using UnityEditor;
using UnityEngine;
using static Algorythm;

[Serializable]
public class Optimizer : EditorWindow
{
    private static Optimizer _intance;
    private static Sprite _sprite;
    private static Vector2Int _resolution = new Vector2Int(8, 8);
    private static int _areaFreshmentSpan = 10;
    private static int _areasVolatilityRange = 100;
    private static PickySizingConfigurator.PickynessLevel _pickinessLevel;
    private static ComputeMode _computeMode;
    private static string _resultFileName = "Assets/scavenger.asset";

    private static string _numpyFileName = "Py/info/1.npy";
    private static string _cudaFileName = "Cuda/info/data.bytes";

    [MenuItem("Optimizer/Optimize")]
    private static void Main()
    {
        _intance = GetWindow<Optimizer>();
    }

    private ProgressReport _operationProgressReport;
    private ProgressReport _overallProgressReport;
    private CancellationTokenSource _cts;

    [Serializable]
    public class StringsArrayContainer
    {
        public string[] Array;
    }

    [Serializable]
    public class StringsArrayContainerArrayContainer
    {
        public StringsArrayContainer[] Array;
    }

    private void OnGUI()
    {
        var newSprite = EditorGUILayout.ObjectField(_sprite, typeof(Sprite), false) as Sprite;
        if (newSprite != _sprite)
            _sprite = newSprite;

        _areaFreshmentSpan = EditorGUILayout.IntField("Areas freshment span:", _areaFreshmentSpan);
        _areasVolatilityRange = EditorGUILayout.IntField("Areas volatility range:", _areasVolatilityRange);
        _resolution = EditorGUILayout.Vector2IntField("Area:", _resolution);
        _pickinessLevel = (PickySizingConfigurator.PickynessLevel)EditorGUILayout.EnumPopup($"Sizings variety level", _pickinessLevel);
        _resultFileName = EditorGUILayout.TextField("Result path:", _resultFileName);
        _computeMode = ComputeMode.Gpu;
        _computeMode = (ComputeMode)EditorGUILayout.EnumPopup($"Compute on", _computeMode);

        if (_sprite != null && _cts == null)
        {
            if (GUILayout.Button("Try"))
            {
                var colorResults = getColors(_sprite);
                var algorithmBulder = new AlgorythmBuilder();
                var pivots = colorResults.sprites.Select(s => new MyVector2Float(s.pivot.x, s.pivot.y)).ToArray();
                var algorythm = algorithmBulder
                    .AddSizingsConfigurator<PickySizingConfigurator>(_pickinessLevel)
                    .AddScoreCounter<DefaultScoreCounter>()
                    .SetAreaEnumerator<DefaultAreaEnumerator>()
                    .SetAreasFreshmentSpan(_areaFreshmentSpan)
                    .SetAreasVolatilityRange(_areasVolatilityRange)
                    .Build(colorResults.colors, pivots, _computeMode);
                _operationProgressReport = algorythm.OperationProgressReport;
                _overallProgressReport = algorythm.OverallProgressReport;
                _cts = new CancellationTokenSource();
                launch(algorythm, colorResults.sprites, colorResults.colors, pivots);
            }
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("TryParse"))
            {
                var fullPath = $"{Application.dataPath.Substring(0, Application.dataPath.Length - "Assets".Length)}{_numpyFileName}";
                var bytes = File.ReadAllBytes(fullPath);
                var arr = NumpySerializer.Deserialize(bytes);
                var bytesAgain = NumpySerializer.Serialize(arr);
                Assert.AreEqual(bytes, bytesAgain, "Serialization fail 1");
            }
            if (GUILayout.Button("SendToPython"))
            {
                var colorsResults = getColors(_sprite);
                var sizings = default(IEnumerable<MyVector2>);
                sizings = new PickySizingConfigurator(_pickinessLevel).ConfigureSizings(sizings, colorsResults.colors.Length, _resolution.x, _resolution.y, default);
                var sizingsDeconstructed = sizings.Select(s => new int[] { s.X, s.Y }).ToArray();
                var fullPath = $"{Application.dataPath.Substring(0, Application.dataPath.Length - "Assets".Length)}{_numpyFileName}";
                var fullPathToDirectory = Directory.GetParent(fullPath).ToString();

                for (int i = 0; i < colorsResults.bytes.Length; i++)
                {
                    var currentBytes = colorsResults.bytes[i];
                    var currentSprite = colorsResults.sprites[i];
                    var pathToSprite = AssetDatabase.GetAssetPath(currentSprite);
                    var npyFileName = $"{i}.npy";
                    fullPath = Path.Combine(fullPathToDirectory, npyFileName);
                    var mdArray = currentBytes.ConvertToMultiDimentional();
                    var bytes = NumpySerializer.Serialize(mdArray);
                    File.WriteAllBytes(fullPath, bytes);
                    var arr = NumpySerializer.Deserialize(bytes);
                    Assert.AreEqual(arr, mdArray, "Serialization fail 2");

                    if (i == 0)
                    {
                        var firstUChar = bytes[0];
                        var lengthOfHeader = BitConverter.ToInt16(bytes, 8);
                        var firstUCharOfHeader = bytes[8];
                        var lastChar = bytes[bytes.Length - 1];

                        var byteOfData1 = bytes[10 + lengthOfHeader];
                        var byteOfData2 = bytes[10 + lengthOfHeader + 1];
                        var byteOfData3 = bytes[10 + lengthOfHeader + 2];
                        var byteOfData4 = bytes[10 + lengthOfHeader + 3];
                        Debug.LogError($"firstUChar = {firstUChar}, lengthOfHeader = {lengthOfHeader}, firstUCharOfHeader = {firstUCharOfHeader}, lastChar = {lastChar}. byteOfData1 = {byteOfData1}, byteOfData2 = {byteOfData2}, byteOfData3 = {byteOfData3}, byteOfData4 = {byteOfData4}");
                    }

                    var pathToSpriteInfo = new List<string>();
                    pathToSpriteInfo.Add(pathToSprite);
                    pathToSpriteInfo.Add(currentSprite.name);
                    var spriteReferenceFileName = $"{i}.txt";
                    fullPath = Path.Combine(fullPathToDirectory, spriteReferenceFileName);
                    File.WriteAllLines(fullPath, pathToSpriteInfo);
                }

                fullPath = Path.Combine(fullPathToDirectory, $"sizings.npy");
                var sizingsDeconstructedMD = sizingsDeconstructed.ConvertToMultiDimentional();
                var sizingsBytes = NumpySerializer.Serialize(sizingsDeconstructedMD);
                File.WriteAllBytes(fullPath, sizingsBytes);
                var arr2 = NumpySerializer.Deserialize(sizingsBytes);
                Assert.AreEqual(arr2, sizingsDeconstructedMD, "Serialization fail 3");
            }
            GUILayout.EndHorizontal();
        }

        if (_sprite != null && GUILayout.Button("Send to CUDA"))
        {
            var colorsResults = getColors(_sprite);
            var sizings = default(IEnumerable<MyVector2>);
            sizings = new PickySizingConfigurator(_pickinessLevel).ConfigureSizings(sizings, colorsResults.colors.Length, _resolution.x, _resolution.y, default);
            var sizingsList = sizings.ToList();
            var sizingsDeconstructed = sizings.Select(s => new short[] { (short)s.X, (short)s.Y }).ToArray();


            var dataList = new List<byte>();
            var registry = new List<registryStruct>();

            var spritesCount = (short)colorsResults.bytes.Length;


            var metaList = new List<List<string>>();
            for (int i = 0; i < spritesCount; i++)
            {
                var currentSprite = colorsResults.sprites[i];
                var pathToSprite = AssetDatabase.GetAssetPath(currentSprite);
                var pathToSpriteInfo = new List<string>();
                pathToSpriteInfo.Add(pathToSprite);
                pathToSpriteInfo.Add(currentSprite.name);
                metaList.Add(pathToSpriteInfo);
            }

            var meta = new StringsArrayContainerArrayContainer();
            meta.Array = new StringsArrayContainer[metaList.Count];
            for (int i = 0; i < metaList.Count; i++)
            {
                meta.Array[i] = new StringsArrayContainer();
                meta.Array[i].Array = metaList[i].ToArray();
            }


            var parallelizedSizingsList = new List<byte>();
            for (int i = 0; i < sizingsDeconstructed.Length; i++)
                parallelizedSizingsList.AddRange(BitConverter.GetBytes(sizingsDeconstructed[i][0]));
            for (int i = 0; i < sizingsDeconstructed.Length; i++)
                parallelizedSizingsList.AddRange(BitConverter.GetBytes(sizingsDeconstructed[i][1]));


            var byteLineLength = 0;
            for (int i = 0; i < spritesCount; i++)
            {
                var currentSpriteBytes = colorsResults.bytes[i];
                var width = currentSpriteBytes.Length;
                var height = currentSpriteBytes[0].Length;

                var widthAndHeight = width << 16 | height;
                var registryEntry = new registryStruct();
                registryEntry.SpritesByteOffset = dataList.Count;
                //registryEntry.SpritesBitOffset = dataList.Count / 8 + (dataList.Count % 8 == 0 ? 0 : 1);
                registryEntry.WidthAndHeight = widthAndHeight;
                registry.Add(registryEntry);

                byteLineLength += width * height;

                //var aopjkfop = 0;
                for (int x = 0; x < width; x++)
                    for (int y = 0; y < height; y++)
                    {
                        var val = currentSpriteBytes[x][y][0];
                        //if (aopjkfop < 240)
                        //    Debug.Log($"_r({aopjkfop++}): {val}");
                        dataList.Add(val);
                    }
            }

            Debug.Log($"byteLineLength = {byteLineLength}");

            for (int i = 0; i < spritesCount; i++)
            {
                var currentSpriteBytes = colorsResults.bytes[i];
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
                var currentSpriteBytes = colorsResults.bytes[i];
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
                var currentSpriteBytes = colorsResults.bytes[i];
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

                var currentSpriteBytes = colorsResults.bytes[i];
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
                            writeBit(newSpriteSizingVoidMap, bitsCounter++, isVoid ? 0 : 1);

                            //if (i == 0 && j == 18)
                            //    Debug.Log($"	{currentVoidOffset + newSpriteSizingVoidMap.Count}: void ({x}, {y}): {(isVoid ? 0 : 1)} - {newSpriteSizingVoidMap[newSpriteSizingVoidMap.Count - 1]}");
                        }
                    }

                    currentVoidOffset += newSpriteSizingVoidMap.Count;
                    voidMapsLength += newSpriteSizingVoidMap.Count;
                    newSpriteVoidMaps.Add(newSpriteSizingVoidMap);

                    //newSpriteVoidMapsOffsets.Add(currentVoidOffset + newSpriteVoidMaps.Count);
                    if (i == 7 && j == 18)
                        Debug.Log($"Area offset for {i}th sprite and {j}th sizing is {newSpriteVoidMapsOffsets[j]}");
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

                var currentSpriteBytes = colorsResults.bytes[i];
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
                        writeBit(newRFlagsList, bitsCounter, r > 127 ? 1 : 0);
                        writeBit(newGFlagsList, bitsCounter, g > 127 ? 1 : 0);
                        writeBit(newBFlagsList, bitsCounter, b > 127 ? 1 : 0);
                        writeBit(newAFlagsList, bitsCounter, a > 127 ? 1 : 0);

                        //if (i == 7)
                        //    Debug.Log($"for pixel {bitsCounter} ({x}, {y}) the flags of r and g are ({(r > 127 ? 1 : 0)}, {(g > 127 ? 1 : 0)})");
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
                //registryEntry.SpritesBitOffset = flagsLineLength;
                //Debug.Log($"registryEntry.SpritesBitOffset = {registryEntry.SpritesBitOffset}");
                registry[i] = registryEntry;
                flagsLineLength += newRFlagsList.Count;
            }

            Debug.Log($"opaquePixelsCount = {opaquePixelsCount}");

            var registryParalellized = new List<byte>();
            for (int i = 0; i < registry.Count; i++)
                registryParalellized.AddRange(BitConverter.GetBytes(registry[i].SpritesByteOffset));
            //for (int i = 0; i < registry.Count; i++)
            //    registryParalellized.AddRange(BitConverter.GetBytes(registry[i].SpritesBitOffset));
            for (int i = 0; i < registry.Count; i++)
                registryParalellized.AddRange(BitConverter.GetBytes((short)(registry[i].WidthAndHeight >> 16 & 65535)));
            for (int i = 0; i < registry.Count; i++)
                registryParalellized.AddRange(BitConverter.GetBytes((short)(registry[i].WidthAndHeight & 65535)));

            var sizingsCount = (short)sizingsDeconstructed.Length;

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

            //Debug.LogError($"spritesCount = {spritesCount}");

            combinedData[4] = (byte)(sizingsCount & 255);
            combinedData[5] = (byte)(sizingsCount >> 8 & 255);
            //Debug.LogError($"sizingsCount = {sizingsCount}");

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

                    //if (i == 7 && j == 18)
                    //    Debug.Log($"VoidMapOffset = {voidMapsOffsets[i][j]}, first byte of candidateVoidMapGlobal = {voidMaps[i][j]}");
                    //Debug.Log($"void map for i({i}), j({j}): {voidMapsOffsets[i][j]}");
                }
            }

            Debug.Log($"voidBytesCount = {voidMapsLength}");
            combinedData[currentOffset++] = (byte)(voidMapsLength & 255);
            combinedData[currentOffset++] = (byte)(voidMapsLength >> 8 & 255);
            combinedData[currentOffset++] = (byte)(voidMapsLength >> 16 & 255);
            combinedData[currentOffset++] = (byte)(voidMapsLength >> 24 & 255);

            //var ourStartingOffset = 4367;
            //var i2 = 0;
            for (int i = 0; i < voidMaps.Count; i++)
            {
                for (int j = 0; j < voidMaps[i].Count; j++)
                {
                    for (int m = 0; m < voidMaps[i][j].Count; m++)
                    {
                        combinedData[currentOffset++] = voidMaps[i][j][m];
                        //if (i2 >= ourStartingOffset && i2 < ourStartingOffset + 300)
                        //    Debug.Log($"void[{i2 - 1}] = {voidMaps[i][j][m]}");
                        //i2++;
                    }
                }
            }

            Debug.Log($"flagsCount = {flagsLineLength}");
            combinedData[currentOffset++] = (byte)(flagsLineLength & 255);
            combinedData[currentOffset++] = (byte)(flagsLineLength >> 8 & 255);
            combinedData[currentOffset++] = (byte)(flagsLineLength >> 16 & 255);
            combinedData[currentOffset++] = (byte)(flagsLineLength >> 24 & 255);

            var i2 = 0;
            for (int i = 0; i < rFlags.Count; i++)
            {
                for (int j = 0; j < rFlags[i].Count; j++)
                {
                    combinedData[currentOffset++] = rFlags[i][j];
                    //if (i2++ < 1024)
                    //    Debug.Log($"rgbaFlags[{i2 - 1}] = {rFlags[i][j]}");
                }
            }

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
                var currentSpriteBytes = colorsResults.colors[i];
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

            //Debug.Log($"sizings length = {parallelizedSizingsList.Count}");
            //Debug.Log($"registry length = {registryParalellized.Count}");
            //Debug.Log($"data length = {dataList.Count}");

            //for (int i = 0; i < 16; i++)
            //    Debug.Log($"R flag {i}: {rFlags[0][i]}");
            //for (int i = 0; i < 16; i++)
            //    Debug.Log($"R flag {i}: {gFlags[0][i]}");
            //for (int i = 0; i < 16; i++)
            //    Debug.Log($"R flag {i}: {bFlags[0][i]}");
            //for (int i = 0; i < 16; i++)
            //    Debug.Log($"R flag {i}: {aFlags[0][i]}");

            //var parallelizedSizingsArray = parallelizedSizingsList.ToArray();
            //Debug.Log($"Sizings length: {parallelizedSizingsArray.Length}");
            //Debug.Log($"Sizings[0]: {BitConverter.ToInt16(parallelizedSizingsArray, 0)}");
            //Debug.Log($"Sizings[1]: {BitConverter.ToInt16(parallelizedSizingsArray, 2)}");
            //Debug.Log($"Sizings[2]: {BitConverter.ToInt16(parallelizedSizingsArray, 4)}");
            //Debug.Log($"Sizings[3]: {BitConverter.ToInt16(parallelizedSizingsArray, 6)}");

            //var registryParalellizedArray = registryParalellized.ToArray();
            //Debug.Log($"Registry length: {registryParalellizedArray.Length}");
            //Debug.Log($"Registry[0]: {BitConverter.ToInt16(registryParalellizedArray, 0)}");
            //Debug.Log($"Registry[1]: {BitConverter.ToInt16(registryParalellizedArray, 4)}");
            //Debug.Log($"Registry[2]: {BitConverter.ToInt16(registryParalellizedArray, 8)}");
            //Debug.Log($"Registry[3]: {BitConverter.ToInt16(registryParalellizedArray, 12)}");

            //var dataArray = dataList.ToArray();
            //Debug.Log($"Data length: {dataArray.Length}");
            //Debug.Log($"Data[0]: {dataArray[0]}");
            //Debug.Log($"Data[1]: {dataArray[1]}");
            //Debug.Log($"Data[2]: {dataArray[2]}");
            //Debug.Log($"Data[3]: {dataArray[3]}");
            //Debug.Log($"Data[4]: {dataArray[4]}");
            //Debug.Log($"Data[5]: {dataArray[5]}");
            //Debug.Log($"Data[6]: {dataArray[6]}");
            //Debug.Log($"Data[7]: {dataArray[7]}");
            //Debug.Log($"Data[8]: {dataArray[8]}");
            //Debug.Log($"Data[9]: {dataArray[9]}");
            //Debug.Log($"Data[10]: {dataArray[10]}");
            //Debug.Log($"Data[11]: {dataArray[11]}");
            //Debug.Log($"Data[12]: {dataArray[12]}");
            //Debug.Log($"Data[13]: {dataArray[13]}");
            //Debug.Log($"Data[14]: {dataArray[14]}");
            //Debug.Log($"Data[15]: {dataArray[15]}");

            var metaTextBytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(meta));
            var metaLength = metaTextBytes.Length;
            var header = new byte[metaLength + 4];
            header[0] = (byte)(metaLength & 255);
            header[1] = (byte)(metaLength >> 8 & 255);
            header[2] = (byte)(metaLength >> 16 & 255);
            header[3] = (byte)(metaLength >> 24 & 255);
            Buffer.BlockCopy(metaTextBytes, 0, header, 4, metaLength);
            //Debug.LogError($"metaLength = {metaLength}");

            var finalBlob = new byte[header.Length + combinedData.Length];
            Buffer.BlockCopy(header, 0, finalBlob, 0, header.Length);
            Buffer.BlockCopy(combinedData, 0, finalBlob, header.Length, combinedData.Length);


            var combinedDataOffset = header.Length;
            var sizingsOffset = combinedDataOffset + 6;
            var registryOffset = sizingsOffset + parallelizedSizingsList.Count;
            var dataOffset = registryOffset + registryParalellized.Count;

            File.WriteAllBytes(_cudaFileName, finalBlob);
            Debug.Log($"Sent!");

            //var blob = File.ReadAllBytes(_cudaFileName);
            //Debug.LogError($"blob length = {blob.Length}");
            //for (int i = 0; i < registryParalellized.Count; i++)
            //{
            //    Debug.LogError($"for registry {i} (offset: {registryOffset + i}): {blob[registryOffset + i]}");
            //}

            //Тестируем подсчет очков.
            var workingOffsets = new int[spritesCount * sizingsCount];
            var workingArrayLength = 0;
            for (int i = 0; i < spritesCount; i++)
            {
                var spriteWidth = colorsResults.colors[i].Length;
                var spriteHeight = colorsResults.colors[i][0].Length;
                for (int j = 0; j < sizingsCount; j++)
                {
                    var sizingWidth = sizingsList[j].X;
                    var sizingHeight = sizingsList[j].Y;

                    workingOffsets[i * sizingsCount + j] = workingArrayLength;
                    workingArrayLength += (spriteWidth - sizingWidth + 1) * (spriteHeight - sizingHeight + 1);
                }
            }

            var i3 = 0;
            var coincidents = new int[workingArrayLength];
            var coincidentsCounts = new int[workingArrayLength];
            var coincidentsSpriteIndexes = new int[workingArrayLength];
            var coincidentsSpriteX = new int[workingArrayLength];
            var coincidentsSpriteY = new int[workingArrayLength];
            var coincidentsSpriteWidth = new int[workingArrayLength];
            var coincidentsSpriteHeight = new int[workingArrayLength];

            for (int i = 0; i < colorsResults.colors.Length; i++)
            {
                var ourColors = colorsResults.colors[i];
                for (int s = 0; s < sizingsCount; s++)
                {
                    var sizing = sizings.ToList()[s];
                    for (int j = 0; j < colorsResults.colors.Length; j++)
                    {
                        var candidateColors = colorsResults.colors[j];
                        for (int ourX = 0; ourX < ourColors.Length - sizing.X + 1; ourX++)
                        {
                            for (int ourY = 0; ourY < ourColors[ourX].Length - sizing.Y + 1; ourY++)
                            {
                                //if (i == 0 && s == 0)
                                //if (i3++ < 240)
                                //Debug.Log($"r({i3++}): {ourColors[ourX][ourY].R}");

                                var score = 0;

                                for (int x = 0; x < sizing.X; x++)
                                {
                                    for (int y = 0; y < sizing.Y; y++)
                                    {
                                        var ourPixelX = ourX + x;
                                        var ourPixelY = ourY + y;
                                        var ourPixel = ourColors[ourPixelX][ourPixelY];
                                        if (ourPixel.A != 0)
                                            score++;
                                    }
                                }
                                //if (i3++ < 100)
                                //    Debug.LogError($"score = {score}");

                                var coincidentsIndex = workingOffsets[i * sizingsCount + s] + ourX * (ourColors[ourX].Length - sizing.Y + 1) + ourY;

                                var currentCoincidents = 0;
                                for (int candidateX = 0; candidateX < candidateColors.Length - sizing.X + 1; candidateX++)
                                {
                                    for (int candidateY = 0; candidateY < candidateColors[candidateX].Length - sizing.Y + 1; candidateY++)
                                    {
                                        var theSame = true;
                                        for (int x = 0; x < sizing.X; x++)
                                        {
                                            for (int y = 0; y < sizing.Y; y++)
                                            {
                                                var ourPixelX = ourX + x;
                                                var ourPixelY = ourY + y;
                                                var candidatePixelX = candidateX + x;
                                                var candidatePixelY = candidateY + y;

                                                var ourPixel = ourColors[ourPixelX][ourPixelY];
                                                var candidatePixel = candidateColors[candidatePixelX][candidatePixelY];

                                                if (ourPixel.R != candidatePixel.R)
                                                {
                                                    theSame = false;
                                                    break;
                                                }
                                                if (ourPixel.G != candidatePixel.G)
                                                {
                                                    theSame = false;
                                                    break;
                                                }
                                                if (ourPixel.B != candidatePixel.B)
                                                {
                                                    theSame = false;
                                                    break;
                                                }
                                                if (ourPixel.A != candidatePixel.A)
                                                {
                                                    theSame = false;
                                                    break;
                                                }
                                            }

                                            if (!theSame)
                                                break;
                                        }
                                        if (theSame)
                                        {
                                            currentCoincidents++;
                                            if (coincidentsIndex == 5140)
                                                Debug.Log($"Coince: {j} ({candidateX},{candidateY})");
                                        }
                                    }
                                }

                                //if (currentCoincidents > 1 && i3++ < 100)
                                //    Debug.LogError($"score = {score}, currentCoincidents = {currentCoincidents}, asokd = {currentCoincidents * score}");
                                if (coincidentsIndex == 5140)
                                {
                                    Debug.Log($"{coincidentsIndex}: {currentCoincidents} coincidents, opaque = {score}, square = {sizing.X * sizing.Y}, score = {(score * score * score / (sizing.X * sizing.Y))}");
                                }
                                //if (coincidentsIndex == 4690)
                                //    Debug.LogError($"WINNER score = {score}, currentCoincidents = {currentCoincidents}, asokd = {currentCoincidents * score}");
                                coincidentsCounts[coincidentsIndex] += currentCoincidents;
                                coincidents[coincidentsIndex] += currentCoincidents * (score * score * score / (sizing.X * sizing.Y));
                                coincidentsSpriteIndexes[coincidentsIndex] = i;
                                coincidentsSpriteX[coincidentsIndex] = ourX;
                                coincidentsSpriteY[coincidentsIndex] = ourY;
                                coincidentsSpriteWidth[coincidentsIndex] = sizing.X;
                                coincidentsSpriteHeight[coincidentsIndex] = sizing.Y;
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < 100; i++)
            {
                Debug.Log($"coincidents[{i}] = {coincidents[i]}");
            }
            for (int i = coincidents.Length - 100; i < coincidents.Length; i++)
            {
                Debug.Log($"coincidents[{i}] = {coincidents[i]}");
            }
            var largestScore = 0;
            var largestScoreIndex = -1;
            for (int i = 0; i < coincidents.Length; i++)
            {
                if (coincidents[i] > largestScore)
                {
                    largestScore = coincidents[i];
                    largestScoreIndex = i;
                }
            }
            Debug.Log($"The Winner is: {largestScoreIndex} with the total score of {largestScore}");
            Debug.Log($"The Winner: sprite {coincidentsSpriteIndexes[largestScoreIndex]}, x {coincidentsSpriteX[largestScoreIndex]}, y {coincidentsSpriteY[largestScoreIndex]}, width {coincidentsSpriteWidth[largestScoreIndex]}, height {coincidentsSpriteHeight[largestScoreIndex]}. Coincidents count = {coincidentsCounts[largestScoreIndex]}");
        }
        if (_cts != null)
        {
            EditorGUILayout.LabelField($"{_overallProgressReport.OperationDescription}: {_overallProgressReport.OperationsDone}/{_overallProgressReport.OperationsCount}");
            EditorGUILayout.LabelField($"Current operation: {_operationProgressReport.OperationDescription} - {_operationProgressReport.OperationsDone}/{_operationProgressReport.OperationsCount}");
            //EditorGUILayout.LabelField($"Progress: {_operationProgressReport.OperationsDone} of {_operationProgressReport.OperationsCount}");
            if (GUILayout.Button("Cancel"))
            {
                _cts.Cancel();
                _cts = null;
                _operationProgressReport = null;
            }
        }

        drawParsingCuda();

        Repaint();
    }

    //Parsing cuda START

    private static TextAsset _importedFromCuda;

    private void drawParsingCuda()
    {
        var newImportedFromCuda = EditorGUILayout.ObjectField(_importedFromCuda, typeof(TextAsset), false) as TextAsset;
        if (newImportedFromCuda != _importedFromCuda)
            _importedFromCuda = newImportedFromCuda;

        if (_importedFromCuda != default && GUILayout.Button("Parse Cuda"))
        {
            var bytes = _importedFromCuda.bytes;
            Debug.LogError($"bytes length = {bytes.Length}");

            var metaLength = BitConverter.ToInt32(bytes, 1);
            Debug.LogError($"metaLength = {metaLength}");

            var metaText = Encoding.UTF8.GetString(bytes, 5, metaLength);
            Debug.Log($"metaText = {metaText}");
        }
    }

    //Parsing cuda FINISH

    private void writeBit(List<byte> bitsContainer, long bitsCounter, int bit)
    {
        var reminder = (int)(bitsCounter % 8);
        if (reminder == 0)
            bitsContainer.Add(0);
        bitsContainer[bitsContainer.Count - 1] |= (byte)((bit & 1) << reminder);
    }

    private async void launch(Algorythm algorythm, Sprite[] sprites, MyColor[][][] colors, MyVector2Float[] pivots)
    {
        await algorythm.Initialize(_resolution, _cts.Token);
        //return;
        var correlationsAndImages = await algorythm.Run();
        var correlations = correlationsAndImages.correlations;

        //var images = correlationsAndImages.testImages;
        //var filesPathBegin = @"C:\Users\celti\Documents\TestImages";
        //for (int i = 0; i < images.Length; i++)
        //{
        //    var tex = new Texture2D(images[i].Width, images[i].Height, TextureFormat.RGBA32, false, false);
        //    tex.filterMode = FilterMode.Point;
        //    tex.SetPixels(images[i].Colors);
        //    tex.Apply();
        //    var fullPath = Path.Combine(filesPathBegin, images[i].filePath);
        //    Directory.CreateDirectory(Directory.GetParent(fullPath).ToString());
        //    File.WriteAllBytes(fullPath, tex.EncodeToPNG());
        //    DestroyImmediate(tex);
        //}
        //Resources.UnloadUnusedAssets();

        var areasPerSprite = getAreasPerSprite(correlations, colors, pivots, sprites.Select(s => s.pixelsPerUnit).ToArray());

        Debug.Log($"correlations.Count = {correlations.Length}");
        Debug.Log($"Максимальное кол-во областей в одном спрайте: {areasPerSprite.OrderByDescending(aps => aps.Value.Count).First().Value.Count}");
        Debug.Log($"Минимальное кол-во областей в одном спрайте: {areasPerSprite.OrderBy(aps => aps.Value.Count).First().Value.Count}");
        Debug.Log($"Общее кол-во непрозрачных пикселей: {correlations.Aggregate(0, (count, cor) => count += cor.Colors.Length, count => count)}");
        Debug.Log($"Общее кол-во ссылок: {correlations.Aggregate(0, (count, cor) => count += cor.Coordinates.Length, count => count)}");

        Debug.Log($"Проверка на кол-во непрозрачных пикселей по спрайтам:");
        var beforeCounts = new int[colors.Length];
        for (int i = 0; i < colors.Length; i++)
        {
            var width = colors[i].Length;
            for (int x = 0; x < width; x++)
            {
                var height = colors[i][x].Length;
                for (int y = 0; y < height; y++)
                {
                    var color = colors[i][x][y];
                    if (color.A > 0)
                        beforeCounts[i]++;
                }
            }
        }

        var afterCounts = new int[colors.Length];
        for (int i = 0; i < colors.Length; i++)
        {
            var spriteChunks = areasPerSprite[i];
            for (int j = 0; j < spriteChunks.Count; j++)
            {
                var currentChunk = spriteChunks[j];
                for (int x = 0; x < currentChunk.Colors.Length; x++)
                {
                    for (int y = 0; y < currentChunk.Colors[x].Length; y++)
                    {
                        var color = currentChunk.Colors[x][y];
                        if (color.A > 0)
                            afterCounts[i]++;
                    }
                }
            }
        }
        for (int i = 0; i < beforeCounts.Length; i++)
        {
            Debug.Log($" #{i + 1}. {beforeCounts[i]} = {afterCounts[i]}");
        }

        _operationProgressReport = null;
        _cts = null;

        //saveSpritesInfo(areasPerSprite, sprites);
        saveCompressedInfo(saveSpritesInfo(areasPerSprite, sprites, pivots, colors.Select(c => new MyVector2Float(c.Length, c[0].Length)).ToArray()));
    }

    private void saveCompressedInfo(UnityOptimizedSpritesStructure unityOptimizedSpritesStructure)
    {
        var packedTextures = new Texture2D(1, 1, TextureFormat.ARGB32, false, false);
        packedTextures.filterMode = FilterMode.Point;
        var allSprites = unityOptimizedSpritesStructure.Chunks.Aggregate(new List<Sprite>(), (list, wr) =>
        {
            list.AddRange(wr.Array.Select(a => a.ChunkSprite));
            return list;
        }, list => list.Distinct().ToArray());
        var atlas = packedTextures.PackTextures(allSprites.Select(s => s.texture).ToArray(), 0);
        var atlasInt = atlas.Select(r => new RectInt(Mathf.FloorToInt(r.xMin * packedTextures.width), Mathf.FloorToInt(r.yMin * packedTextures.height), Mathf.FloorToInt(r.width * packedTextures.width), Mathf.FloorToInt(r.height * packedTextures.height))).ToArray();
        var atlasMaxX = atlasInt.OrderByDescending(r => r.xMin).First().xMin;
        var atlasMaxY = atlasInt.OrderByDescending(r => r.yMin).First().yMin;
        var atlasMaxWidth = atlasInt.OrderByDescending(r => r.width).First().width;
        var atlasMaxHeight = atlasInt.OrderByDescending(r => r.height).First().height;
        var atlasXBits = getBitsCount(atlasMaxX);
        var atlasYBits = getBitsCount(atlasMaxY);
        var atlasWidthBits = getBitsCount(atlasMaxWidth);
        var atlasHeightBits = getBitsCount(atlasMaxHeight);
        /*
         * Ок, такая фишка, что каждый спрайт в кадре будет брать спрайт соответствующий его порядковому номеру, поэтому не нужно будет хранить
         * этот самый порядковый номер. Хотя да, и так и так мы это не храним. А что мы храним?
         * 
         * Нам нужны смещения спрайтов, координаты области и ее размеры. Таким образом, если мы, например, отображаем 9й спрайт, остоящий из 141
         * чанка, то мы сначала ищем смещение, соответствующее 9му спрайту, а там мы уже идем по порядку, расставляя все чанки. На самом деле нам
         * даже не нужно хранить смещение. Достаточно будет как-то понять, что спрайт закончился. О! Еще офигенная тема - не обязательно менять
         * чанки местами. Если чанк №31 содержит спрайт №2 а на следующем кадре спрайт №2 содержит чанк №56, то зачем так делать? можно просто 
         * оставить этот спрайт у чанка 31 на столько насколько потребуется и сэкономить ресурсы и памяти и процессора. Тогда надо будет хранить 
         * инфу по чанкам. Т.е. у нас на все спрайты данного атласа максимум - 141 чанк, и идем дальше по чанкам.
         */

        var maximumChunks = unityOptimizedSpritesStructure.Chunks.OrderByDescending(c => c.Array.Length).First().Array.Length;
        var keyframesCount = unityOptimizedSpritesStructure.Chunks.Length;
        var allChunks = unityOptimizedSpritesStructure.Chunks.Aggregate(new List<SpriteChunk>(), (list, c) =>
        {
            list.AddRange(c.Array);
            return list;
        }, c => c.ToArray());
        var maximumX = allChunks.OrderByDescending(c => c.Area.X).First().Area.X;
        var maximumY = allChunks.OrderByDescending(c => c.Area.Y).First().Area.Y;
        var minimumX = allChunks.OrderBy(c => c.Area.X).First().Area.X;
        var minimumY = allChunks.OrderBy(c => c.Area.Y).First().Area.Y;
        var maximumWidth = allChunks.OrderByDescending(c => c.Area.Width).First().Area.Width;
        var maximumHeight = allChunks.OrderByDescending(c => c.Area.Height).First().Area.Height;

        var atlasIndexBits = getBitsCount(atlas.Length);
        var xBits = getBitsCount(maximumX - minimumX);
        var yBits = getBitsCount(maximumY - minimumY);
        //var widthBits = getBitsCount(maximumWidth);
        //var heightBits = getBitsCount(maximumHeight);
        //var keysrameBits = getBitsCount(keyframesCount);

        _groupsCount.Clear();

        var firstPassStruct = new SpriteChunk[maximumChunks][];
        for (int i = 0; i < maximumChunks; i++)
            firstPassStruct[i] = new SpriteChunk[keyframesCount];
        for (int k = 0; k < keyframesCount; k++)
        {
            var currentChunks = unityOptimizedSpritesStructure.Chunks[k].Array.ToList();

            //Гораздо дешевле хранить инфу о размере, чем о координатах, поэтому мы стараемся, чтобы чанки стояли на местах как можно дольше. О чем 
            //это я? Мы все равно ссылаемся просто на номер чанка на атласе, так что размер... Хотя да, и размер и координаты могут меняться. Только
            //в случае размера меняется только порядковый номер чанка в атласе. В общем, приоритет такой - неподвижность чанка -> неизменность чанка.
            //Т.е. мы всегда стараемся сохранить неподвижность, а уже потом неизменность. Если удается сохранить и то и то - это бесплатный кифрейм.
            //Хотя позиции тоже можно сжать. Если позиции повторяются, на них можно сослаться. Другое дело, что повторов позиций не должно быть полезно 
            //много. С третьей стороны, повторов цифр должно быть гораздо больше, чем повторов координат.

            if (k == 0)
            {
                //В первый кифрейм просто заполняем все как есть - мы ничего не можем улучшить

                for (int c = 0; c < currentChunks.Count; c++)
                    firstPassStruct[c][k] = currentChunks[c];

                continue;
            }

            //Во всех кифреймах после первого пытаемся сэкономить место.

            /*
             * Ок, тут мы не можем идти по порядку, т.к. мы можем случайно забрать у кого-то следующего его идеальный вариант.
             * Поэтому мы проходимся по вариантам. Сначала расставляем все идеальные, потом все хуже, хуже и так до конца. Таким образом
             * у нас могут появиться "дырки" в массиве, так что мы не будем точно знать когда закончить.
             */

            /*
             * В общем, сначала мы должны упорядочить возможные варианты "хорошести" вариантов и пройтись по всем, начиная с лучших, заканчивая
             * худшими или пока не закончатся чанки.
             */

            var lastChunks = unityOptimizedSpritesStructure.Chunks[k - 1].Array.ToList();
            var allScores = new List<int>();
            //for (int i = 0; i < currentChunks.Count; i++)
            //    for (int j = 0; j < lastChunks.Count; j++)
            //        allScores.Add(getScoreOfTwoChunks(currentChunks[i], lastChunks[j], xBits, yBits, atlasIndexBits));
            currentChunks.ForEach(current => lastChunks.ForEach(last => allScores.Add(getScoreOfTwoChunks(current, last, xBits, yBits, atlasIndexBits))));
            var groups = allScores.GroupBy(score => score).Select(gr => gr.Key).OrderBy(score => score).ToList();
            var groupIndex = 0;
            while (currentChunks.Count > 0)
            {
                /*
                 * Т.к. мы сопоставляем все прошлые со всеми нынешнеми и у нас нету default'ов, мы можем остаться без last'ов для всех наших current'ов.
                 * Поэтому остатки означают, что для них не хватило места - запихиваем их куда попало.
                 */
                if (groups.Count == 0)
                    break;
                var currentGroup = groups[0];
                groups.RemoveAt(0);

                var goodOnes = new List<(SpriteChunk current, SpriteChunk last)>();
                for (int i = 0; i < currentChunks.Count; i++)
                {
                    for (int j = 0; j < lastChunks.Count; j++)
                    {
                        //Тут хочется ввернуть проверку, чтобы как-то ограничить кол-во lastChunks только теми, которые возможно продолжить, но тут не все так просто - мы взяли lastChunks не из firstPassStruct, так что индексы не совпадают, а обход этого выглядел бы довольно громоздко, да и черт знает пока как быть с default, так что пока буду keep it simple.
                        if (getScoreOfTwoChunks(currentChunks[i], lastChunks[j], xBits, yBits, atlasIndexBits) <= currentGroup)
                            goodOnes.Add((currentChunks[i], lastChunks[j]));
                    }
                }

                for (int i = 0; i < goodOnes.Count; i++)
                {
                    /*
                     * На самом деле current и last не могут быть default, т.к. мы их взяли из unityOptimizedSpritesStructure, а там нет пустых вхождений.
                     * Пустые вхождения мы получаем, только когда имеем дело с firstPassStruct, т.к. там одинаковое число чанков для каждого кифрейма, равное
                     * максимальному.
                     */
                    var (current, last) = goodOnes[i];

                    //Чтобы проверить, что мы рассуждаем правильно, сотрем пока все варианты где current и last равны default
                    //if (current == default) //Мы ничего не делаем для пустых вхождений
                    //    continue;

                    if (!currentChunks.Contains(current)) //Если currentChunks не содержит его, значит мы этот чанк уже обработали...
                        continue;

                    //if (last == default)
                    //{
                    //    //Если прошлого чанка у нас нет, значит все лучшие решения уже применены и нам можно безопасно взять первый свободный индекс
                    //    var lastChunkIndex = 0;
                    //    while (firstPassStruct[lastChunkIndex][k] != default)
                    //        lastChunkIndex++;

                    //    firstPassStruct[lastChunkIndex][k] = current; //Как-то так.
                    //    currentChunks.Remove(current);
                    //}
                    //else
                    {
                        var lastChunkIndex = 0;
                        for (int j = 0; j < firstPassStruct.Length; j++)
                        {
                            if (firstPassStruct[j][k - 1] == last)
                            {
                                lastChunkIndex = j;
                                break;
                            }
                        }
                        //var lastChunkIndex = lastChunks.IndexOf(last);
                        if (firstPassStruct[lastChunkIndex][k] != default) //Если чанк прошлого кадра заполнен - мы ничего уже не сможем сделать, т.к. в случае успеха мы должны были бы заполнить этот чанк этого кадра. 
                                                                           //А такое вполне может быть, т.к. мы не фильтруем пока прошлые чанки, а сопоставляем все currentChunks (число которых уменьшается) со всеми lastChunks (которые остаются в том же кол-ве).
                            continue;

                        firstPassStruct[lastChunkIndex][k] = current; //Как-то так. Вроде бы...
                        currentChunks.Remove(current);
                    }
                }

                groupIndex++;
            }

            while (currentChunks.Count > 0)
            {
                for (int i = 0; i < maximumChunks; i++)
                {
                    if (firstPassStruct[i][k] == default)
                    {
                        firstPassStruct[i][k] = currentChunks[0];
                        currentChunks.RemoveAt(0);
                        break;
                    }
                }
            }

            //Далее идут legacy-варианты, где я пытался применить не-общее решение:

            ////Сначала пытаемся забрать самые лучшие варианты.
            //for (int c = 0; c < maximumChunks; c++)
            //{
            //    if (currentChunks.Count == 0)
            //        break;

            //    var formerChunk = firstPassStruct[c][k - 1];
            //    if (formerChunk == default) //Без прошлого чанка самый лучший вариант не получится
            //        continue;

            //    //Пытаемся сохранить обе координаты...
            //    var theCoolestChunks = getChunksWithSameCoordinates(currentChunks, formerChunk);
            //    if (!theCoolestChunks.Any())
            //        continue;

            //    theCoolestChunks = getChunksWithSameSprite(theCoolestChunks, formerChunk);
            //    if (!theCoolestChunks.Any())
            //        continue;

            //    firstPassStruct[c][k] = theCoolestChunks.First();
            //    currentChunks.Remove(firstPassStruct[c][k]);
            //}

            //for (int c = 0; c < maximumChunks; c++)
            //{
            //    if (currentChunks.Count == 0)
            //        break;

            //    if (firstPassStruct[c][k] != default)
            //        continue; //Этот позицию мы уже заполнили до этого чем-то лучшим

            //    var formerChunk = firstPassStruct[c][k - 1];
            //    if (formerChunk == default) //Без прошлого чанка второй лучший вариант не получится
            //        continue;
            //}

            //for (int c = 0; c < maximumChunks; c++)
            //{
            //    //Если все чанки данного кифрейма уже распределены - выходим из цикла
            //    if (currentChunks.Count == 0)
            //        break;

            //    var formerChunk = firstPassStruct[c][k - 1];

            //    //Пытаемся сохранить обе координаты...
            //    var theCoolestChunks = getChunksWithSameCoordinates(currentChunks, formerChunk);

            //    //Если обе координаты ни у кого не совпадают - пытаемся сохранить спрайт
            //    if (!theCoolestChunks.Any())
            //        theCoolestChunks = getChunksWithSameSprite(currentChunks, formerChunk);
            //    else
            //    {
            //        //Если нашли - пытаемся еще и чтобы был тот же спрайт
            //        var theMoreCoolChunks = getChunksWithSameSprite(theCoolestChunks, formerChunk);
            //        if (theMoreCoolChunks.Any())
            //            theCoolestChunks = theMoreCoolChunks; //Это победа, дальше не ищем
            //        firstPassStruct[c][k] = theCoolestChunks.First(); //Даже если спрайт не совпадает - лучше мы уже не найдем.
            //        continue;
            //    }

            //    //Если ни у кого спрайт не такой же - пытаемся сохранить хотя бы одну из координат
            //    if (!theCoolestChunks.Any())
            //        theCoolestChunks = getChunksWithAtLeastOneSameCoordinate(currentChunks, formerChunk);
            //    else
            //    {
            //        //Если нашли кого-то с таким же спрайтом - смотрим, может у кого-то еще и хотя бы 1 координата совпадает
            //        var theMoreCoolChunks = getChunksWithAtLeastOneSameCoordinate(theCoolestChunks, formerChunk);
            //        if (theMoreCoolChunks.Any())
            //            theCoolestChunks = theMoreCoolChunks; //Это почти что полностью победа, дальше не ищем
            //        firstPassStruct[c][k] = theCoolestChunks.First(); //Даже если ни одна координата ни у кого не совпадает - лучше мы уже не найдем.
            //        continue;
            //    }

            //    //Ну и если ни у кого даже этого нет - берем любой чанк
            //    if (!theCoolestChunks.Any())
            //        theCoolestChunks = currentChunks;

            //    firstPassStruct[c][k] = theCoolestChunks.First();
            //}
        }

        _xBitsCount = 0;
        _yBitsCount = 0;
        _atlasIndexBitsCount = 0;
        _rectXBitsCount = 0;
        _rectYBitsCount = 0;
        _rectWidthBitsCount = 0;
        _rectHeightBitsCount = 0;
        _allTheFlagsBitsCount = 0;
        _headersBitsCount = 0;
        _defaultCount = 0;
        _nonDefaultCount = 0;

        //Чанки мы упорядочили - осталось только записать
        var secondPassChunks = new List<byte>[maximumChunks];
        for (int i = 0; i < maximumChunks; i++)
            secondPassChunks[i] = new List<byte>();

        for (int c = 0; c < maximumChunks; c++)
        {
            var list = secondPassChunks[c];
            var keyframes = firstPassStruct[c];
            var formerChunk = default(SpriteChunk);
            for (int k = 0; k < keyframes.Length; k++)
            {
                var chunk = keyframes[k];

                writeChunkBytes(allSprites, chunk, formerChunk, list, atlasIndexBits, xBits, yBits);

                formerChunk = chunk;
            }
        }

        //Записываем смещения
        var maxOffsetX = Mathf.FloorToInt(unityOptimizedSpritesStructure.Pivots.OrderBy(p => p.X).First().X);
        var maxOffsetY = Mathf.FloorToInt(unityOptimizedSpritesStructure.Pivots.OrderBy(p => p.Y).First().Y);
        var offsetXBits = getBitsCount(maxOffsetX);
        var offsetYBits = getBitsCount(maxOffsetY);
        var offsetBits = new List<byte>();
        for (int i = 0; i < unityOptimizedSpritesStructure.Pivots.Length; i++)
        {
            var pivot = unityOptimizedSpritesStructure.Pivots[i];
            offsetBits.AddRange(toBits(Mathf.FloorToInt(pivot.X), offsetXBits));
            offsetBits.AddRange(toBits(Mathf.FloorToInt(pivot.Y), offsetYBits));
        }

        var lengthsLength = 16;

        var atlasBits = new List<byte>();

        atlasBits.AddRange(toBits(atlasInt.Length, lengthsLength));
        for (int r = 0; r < atlasInt.Length; r++)
        {
            var rect = atlasInt[r];
            atlasBits.AddRange(toBits(rect.xMin, atlasXBits));
            atlasBits.AddRange(toBits(rect.yMin, atlasYBits));
            atlasBits.AddRange(toBits(rect.width, atlasWidthBits));
            atlasBits.AddRange(toBits(rect.height, atlasHeightBits));
            _rectXBitsCount += atlasXBits;
            _rectYBitsCount += atlasYBits;
            _rectWidthBitsCount += atlasWidthBits;
            _rectHeightBitsCount += atlasHeightBits;
        }

        var allMaxBits = new int[]
            {
                atlasXBits,
                atlasYBits,
                atlasWidthBits,
                atlasHeightBits,
                atlasIndexBits,
                xBits,
                yBits,
                offsetXBits,
                offsetYBits
            };
        var allRelativelyShortSingleValues = new List<int>();
        allRelativelyShortSingleValues.AddRange(allMaxBits);
        //allSingleValues.Add(atlasInt.Length); //Их сюда не включаем, т.к. они должна быть значительно больше,
        //allSingleValues.Add(secondPassChunks.Length); //так что, думаю, уместнее использовать просто 16.
        allRelativelyShortSingleValues.Add(firstPassStruct[0].Length);

        Debug.Log($"atlasXBits = {atlasXBits}");
        Debug.Log($"atlasYBits = {atlasYBits}");
        Debug.Log($"atlasWidthBits = {atlasWidthBits}");
        Debug.Log($"atlasHeightBits = {atlasHeightBits}");
        Debug.Log($"atlasIndexBits = {atlasIndexBits}");
        Debug.Log($"xBits = {xBits}");
        Debug.Log($"yBits = {yBits}");
        var maxMaxBits = allRelativelyShortSingleValues.OrderByDescending(v => v).First();
        var maxMaxBitsBits = getBitsCount(maxMaxBits);

        var header = new List<byte>();
        header.AddRange(toBits(maxMaxBitsBits, lengthsLength));//Для этого значения резервируем 2 байта, т.к. вряд ли какая-то величина будет когда-то весить больше
        _headersBitsCount += lengthsLength;

        for (int i = 0; i < allMaxBits.Length; i++)
        {
            header.AddRange(toBits(allMaxBits[i], maxMaxBitsBits));
            _headersBitsCount += maxMaxBitsBits;
        }

        var everyBitAsBytes = new List<byte>();
        everyBitAsBytes.AddRange(header);
        everyBitAsBytes.AddRange(atlasBits);
        everyBitAsBytes.AddRange(toBits(secondPassChunks.Length, lengthsLength));
        everyBitAsBytes.AddRange(toBits(firstPassStruct[0].Length, maxMaxBitsBits));
        everyBitAsBytes.AddRange(offsetBits);
        for (int i = 0; i < secondPassChunks.Length; i++)
            everyBitAsBytes.AddRange(secondPassChunks[i]);

        var resultBytes = new byte[Mathf.CeilToInt(everyBitAsBytes.Count / 8f)];
        for (int i = 0; i < resultBytes.Length; i++)
        {
            var currentByteAsInt = 0;
            for (int b = 0; b < 8; b++)
            {
                var bitIndex = i * 8 + b;
                if (bitIndex >= everyBitAsBytes.Count)
                    break;

                var currentBitByte = everyBitAsBytes[bitIndex];
                currentByteAsInt |= currentBitByte << b;
            }
            resultBytes[i] = (byte)currentByteAsInt;
        }

        var fullPath = $"{Application.dataPath.Substring(0, Application.dataPath.Length - "Assets".Length)}{_resultFileName}";
        var fileNameWithoutExtension = Path.GetFileNameWithoutExtension(fullPath);
        var fullPathWithoutExtension = Path.Combine(Directory.GetParent(fullPath).ToString(), fileNameWithoutExtension);
        var atlasFullPath = $"{fullPathWithoutExtension}.png";
        var infoFullPath = $"{fullPathWithoutExtension}.bytes";

        var pngBytes = packedTextures.EncodeToPNG();
        File.WriteAllBytes(atlasFullPath, pngBytes);
        File.WriteAllBytes(infoFullPath, resultBytes);

        Debug.Log($"_defaultCount = {_defaultCount}");
        Debug.Log($"_nonDefaultCount = {_nonDefaultCount}");
        Debug.Log($"_headersBitsCount = {_headersBitsCount}");
        Debug.Log($"_allTheFlagsBitsCount = {_allTheFlagsBitsCount}");
        Debug.Log($"_xBitsCount = {_xBitsCount}");
        Debug.Log($"_yBitsCount = {_yBitsCount}");
        Debug.Log($"_atlasIndexBitsCount = {_atlasIndexBitsCount}");
        Debug.Log($"_rectXBitsCount = {_rectXBitsCount}");
        Debug.Log($"_rectYBitsCount = {_rectYBitsCount}");
        Debug.Log($"_rectWidthBitsCount = {_rectWidthBitsCount}");
        Debug.Log($"_rectHeightBitsCount = {_rectHeightBitsCount}");
        foreach (var kvp in _groupsCount)
            Debug.Log($"group {kvp.Key}: {kvp.Value}");

        Debug.Log($"total length png = {pngBytes.Length}");
        Debug.Log($"total lenfth info = {resultBytes.Length}");

        var blobPlayer = new BlobPlayer(resultBytes);
    }

    private int _headersBitsCount;
    private int _allTheFlagsBitsCount;
    private int _xBitsCount;
    private int _yBitsCount;
    private int _atlasIndexBitsCount;
    private int _rectXBitsCount;
    private int _rectYBitsCount;
    private int _rectWidthBitsCount;
    private int _rectHeightBitsCount;
    private int _defaultCount;
    private int _nonDefaultCount;

    private Dictionary<int, int> _groupsCount = new Dictionary<int, int>();

    private int getScoreOfTwoChunks(SpriteChunk currentChunk, SpriteChunk lastChunk, int xBits, int yBits, int atlasIndexBits)
    {
        //if (currentChunk == default && lastChunk == default)
        //    return int.MaxValue;
        if (currentChunk == default || lastChunk == default)
            return xBits + yBits + atlasIndexBits;

        var lastX = lastChunk.Area.X;
        var lastY = lastChunk.Area.Y;
        var lastSprite = lastChunk.ChunkSprite;
        var currentX = currentChunk.Area.X;
        var currentY = currentChunk.Area.Y;
        var currentSprite = currentChunk.ChunkSprite;

        var score = 0;
        if (lastX != currentX)
            score += xBits;
        if (lastY != currentY)
            score += yBits;
        if (lastSprite != currentSprite)
            score += atlasIndexBits;

        //if (!_groupsCount.ContainsKey(score))
        //    _groupsCount.Add(score, 0);
        //_groupsCount[score]++;

        return score;
    }

    private void writeChunkBytes(Sprite[] sprites, SpriteChunk chunk, SpriteChunk formerChunk, List<byte> list, int atlasIndexBits, int xBits, int yBits)
    {
        if (chunk == default)
        {
            list.Add(0);
            _allTheFlagsBitsCount++;
            _defaultCount++;
            return;
        }
        _nonDefaultCount++;
        list.Add(1);
        _allTheFlagsBitsCount++;

        var spriteFull = (byte)0;
        if (formerChunk == default || formerChunk.ChunkSprite != chunk.ChunkSprite)
            spriteFull = 1;
        var xFull = (byte)0;
        if (formerChunk == default || formerChunk.Area.X != chunk.Area.X)
            xFull = 1;
        var yFull = (byte)0;
        if (formerChunk == default || formerChunk.Area.Y != chunk.Area.Y)
            yFull = 1;
        _allTheFlagsBitsCount++;
        _allTheFlagsBitsCount++;
        _allTheFlagsBitsCount++;

        list.Add(spriteFull);
        list.Add(xFull);
        list.Add(yFull);

        var score = 0;
        if (spriteFull > 0)
        {
            list.AddRange(toBits(Array.IndexOf(sprites, chunk.ChunkSprite), atlasIndexBits));
            _atlasIndexBitsCount += atlasIndexBits;
            score += atlasIndexBits;
        }
        if (xFull > 0)
        {
            list.AddRange(toBits(chunk.Area.X, xBits));
            _xBitsCount += xBits;
            score += xBits;
        }
        if (yFull > 0)
        {
            list.AddRange(toBits(chunk.Area.Y, yBits));
            _yBitsCount += yBits;
            score += yBits;
        }
        if (!_groupsCount.ContainsKey(score))
            _groupsCount.Add(score, 0);
        _groupsCount[score]++;
    }

    private IEnumerable<byte> toBits(int v, int bitsLen)
    {
        var result = new byte[bitsLen];

        for (int i = 0; i < bitsLen; i++)
            result[i] = (byte)(v >> i & 1);

        return result;
    }

    private IEnumerable<byte> toBitsWithSign(int v, int bitsLen)
    {
        var result = new byte[bitsLen];
        var absV = Mathf.Abs(v);
        var sign = v > 0 ? 1 : 0;

        result[0] = (byte)sign;

        for (int i = 1; i < bitsLen; i++)
            result[i] = (byte)(absV >> (i - 1) & 1);

        return result;
    }

    private IEnumerable<SpriteChunk> getChunksWithSameCoordinates(IEnumerable<SpriteChunk> chunks, SpriteChunk chunk) => chunks.Where(c => c.Area.X == chunk.Area.X && c.Area.Y == chunk.Area.Y);
    private IEnumerable<SpriteChunk> getChunksWithSameSprite(IEnumerable<SpriteChunk> chunks, SpriteChunk chunk) => chunks.Where(c => c.ChunkSprite == chunk.ChunkSprite);
    private IEnumerable<SpriteChunk> getChunksWithAtLeastOneSameCoordinate(IEnumerable<SpriteChunk> chunks, SpriteChunk chunk) => chunks.Where(c => c.Area.X == chunk.Area.X || c.Area.Y == chunk.Area.Y);

    private int getBitsCount(int number)
    {
        if (number < 2)
            return 1;
        if (number < 4)
            return 2;
        if (number < 8)
            return 3;
        if (number < 16)
            return 4;
        if (number < 32)
            return 5;
        if (number < 64)
            return 6;
        if (number < 128)
            return 7;
        if (number < 256)
            return 8;
        if (number < 512)
            return 9;
        if (number < 1024)
            return 10;
        if (number < 2048)
            return 11;
        if (number < 4096)
            return 12;
        if (number < 8192)
            return 13;
        if (number < 16_384)
            return 14;
        if (number < 32_768)
            return 15;
        if (number < 65536)
            return 16;
        if (number < 131072)
            return 17;
        if (number < 262144)
            return 18;
        if (number < 524288)
            return 19;
        if (number < 1048576)
            return 20;
        if (number < 2097152)
            return 21;
        if (number < 4194304)
            return 22;
        if (number < 8388608)
            return 23;
        if (number < 16777216)
            return 24;
        if (number < 33554432)
            return 25;
        if (number < 67108864)
            return 26;
        if (number < 134217728)
            return 27;
        if (number < 268435456)
            return 28;
        if (number < 536870912)
            return 29;
        if (number < 1073741824)
            return 30;
        //if (number < 2_147_483_648) //int range exceeded
        return 31;
        //return 32;
    }

    private UnityOptimizedSpritesStructure saveSpritesInfo(Dictionary<int, List<SpriteChunk>> areasPerSprite, Sprite[] sprites, MyVector2Float[] pivots, MyVector2Float[] spriteSizes)
    {
        var newSpritesInfo = CreateInstance<UnityOptimizedSpritesStructure>();
        AssetDatabase.CreateAsset(newSpritesInfo, _resultFileName);
        AssetDatabase.CreateFolder(Path.GetDirectoryName(_resultFileName), Path.GetFileNameWithoutExtension(_resultFileName));
        var spritesDirectory = Path.Combine(Path.GetDirectoryName(_resultFileName), Path.GetFileNameWithoutExtension(_resultFileName));

        newSpritesInfo.Sprites = sprites;
        newSpritesInfo.Pivots = pivots;
        newSpritesInfo.Sizes = spriteSizes;

        var references = new Dictionary<MySerializableColor[][], (ColorsReference reference, Sprite sprite)>();

        var chunks = new SpriteChunkArrayWrapper[sprites.Length];
        foreach (var kvp in areasPerSprite)
        {
            var spriteIndex = kvp.Key;
            var currentChunks = kvp.Value;
            var currentChunksArray = currentChunks.ToArray();

            for (int i = 0; i < currentChunksArray.Length; i++)
            {
                var currentColors = currentChunksArray[i].Colors;
                if (!references.ContainsKey(currentColors))
                {
                    var newColorsReference = ScriptableObject.CreateInstance<ColorsReference>();
                    newColorsReference.Colors = currentColors;

                    var width = currentColors.Length;
                    var height = currentColors[0].Length;
                    var newTexture = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
                    newTexture.filterMode = FilterMode.Point;
                    for (int x = 0; x < width; x++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            var currentColor = currentColors[x][y];
                            var color = new Color(currentColor.R / 255f, currentColor.G / 255f, currentColor.B / 255f, currentColor.A / 255f);
                            newTexture.SetPixel(x, y, color);
                            //Debug.Log($"color = {color}");
                        }
                    }
                    var texturePath = Path.Combine(spritesDirectory, $"Sprite_{references.Count}.png");
                    File.WriteAllBytes(texturePath, newTexture.EncodeToPNG());
                    AssetDatabase.ImportAsset(texturePath);
                    var ti = AssetImporter.GetAtPath(texturePath) as TextureImporter;
                    ti.textureType = TextureImporterType.Sprite;
                    ti.filterMode = FilterMode.Point;
                    ti.alphaIsTransparency = true;
                    ti.mipmapEnabled = false;
                    ti.spriteImportMode = SpriteImportMode.Single;
                    ti.spritePivot = Vector2.down + Vector2.right;
                    ti.isReadable = true;
                    ti.crunchedCompression = false;
                    ti.textureCompression = TextureImporterCompression.Uncompressed;

                    var texSettings = new TextureImporterSettings();
                    ti.ReadTextureSettings(texSettings);
                    texSettings.spriteAlignment = (int)SpriteAlignment.BottomLeft;
                    ti.SetTextureSettings(texSettings);

                    AssetDatabase.ImportAsset(texturePath);
                    var sprite = AssetDatabase.LoadAssetAtPath<Sprite>(texturePath);

                    references.Add(currentColors, (newColorsReference, sprite));

                    AssetDatabase.AddObjectToAsset(newColorsReference, _resultFileName);
                }
                currentChunksArray[i].ColorsReference = references[currentColors].reference;
                currentChunksArray[i].ChunkSprite = references[currentColors].sprite;
                currentChunksArray[i].Colors = default;
            }

            chunks[spriteIndex] = new SpriteChunkArrayWrapper(currentChunksArray);
        }

        newSpritesInfo.Chunks = chunks;

        AssetDatabase.SaveAssets();

        //packAndCreateSpritesForEachReference(references.Values.ToArray());
        return newSpritesInfo;
    }

    //private void packAndCreateSpritesForEachReference(ColorsReference[] colorsReferences)
    //{
    //    Debug.Log($"Trying to pack and create {colorsReferences.Length} sprites");
    //    throw new NotImplementedException();
    //}

    private Dictionary<int, List<SpriteChunk>> getAreasPerSprite(Algorythm.Correlation[] correlations, MyColor[][][] colors, MyVector2Float[] pivots, float[] pixelPerUnits)
    {
        var result = new Dictionary<int, List<SpriteChunk>>();

        for (int i = 0; i < correlations.Length; i++)
        {
            for (int j = 0; j < correlations[i].Coordinates.Length; j++)
            {
                var info = correlations[i].Coordinates[j];
                //var pivot = pivots[info.SpriteIndex];
                //var width = colors[info.SpriteIndex].Length;
                //var height = colors[info.SpriteIndex][0].Length;
                //var ppu = pixelPerUnits[info.SpriteIndex];
                //var offsetX = Mathf.FloorToInt(pivot.X/* * width*//* / ppu*/);
                //var offsetY = Mathf.FloorToInt(pivot.Y/* * height*//* / ppu*/);
                //Debug.LogError($"width = {width}. offsetX = {offsetX}. height = {height}. offsetY = {offsetY}. info.X = {info.X}, info.Y = {info.Y}");
                var pivotedInfo = new MyAreaCoordinates(info.SpriteIndex, info.X/* - offsetX*/, info.Y/* - offsetY*/, info.Width, info.Height);
                if (!result.ContainsKey(info.SpriteIndex))
                    result.Add(info.SpriteIndex, new List<SpriteChunk>());

                result[info.SpriteIndex].Add(new SpriteChunk(correlations[i].Colors, pivotedInfo));
            }
        }

        return result;
    }

    private (byte[][][][] bytes, Sprite[] sprites) getBytes(Sprite sprite)
    {
        var texture = sprite.texture;
        var path = AssetDatabase.GetAssetPath(sprite);
        var allAssetsAtPath = AssetDatabase.LoadAllAssetsAtPath(path);
        var allSptitesAtPath = allAssetsAtPath.OfType<Sprite>().ToArray();
        var ti = AssetImporter.GetAtPath(path) as TextureImporter;
        var fullPath = $"{Application.dataPath.Substring(0, Application.dataPath.Length - "Assets".Length)}{path}";
        //Debug.LogError($"path = {fullPath}");
        texture = new Texture2D(1, 1, TextureFormat.ARGB32, false);
        texture.filterMode = FilterMode.Point;
        texture.LoadImage(File.ReadAllBytes(fullPath));
        //return null;
        ti.isReadable = true;
        ti.SaveAndReimport();

        var spritesCount = ti.spritesheet.Length;
        var bytes = default(byte[][][][]);
        var sprites = default(Sprite[]);
        var sb = new StringBuilder();
        if (spritesCount == 0) //If there're no items in spritesheet - it means there is a single sprite in asset.
        {
            bytes = new byte[1][][][];
            sprites = new Sprite[1];
            sprites[0] = sprite;

            var tex = sprite.texture;
            var currentBytes = new byte[tex.width][][];
            for (int x = 0; x < tex.width; x++)
            {
                currentBytes[x] = new byte[tex.height][];
                for (int y = 0; y < tex.height; y++)
                {
                    var color = texture.GetPixel(x, y);
                    currentBytes[x][y] = new byte[4];
                    currentBytes[x][y][0] = Convert.ToByte(Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue));
                    currentBytes[x][y][1] = Convert.ToByte(Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue));
                    currentBytes[x][y][2] = Convert.ToByte(Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue));
                    currentBytes[x][y][3] = Convert.ToByte(Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue));
                }
            }
            bytes[0] = currentBytes;
        }
        else
        {
            bytes = new byte[spritesCount][][][];
            sprites = new Sprite[spritesCount];

            for (int i = 0; i < spritesCount; i++)
            {
                var currentSprite = ti.spritesheet[i];
                sprites[i] = allSptitesAtPath.Where(s => s.name == currentSprite.name).First();

                var xOrigin = Mathf.FloorToInt(currentSprite.rect.x);
                var yOrigin = Mathf.CeilToInt(currentSprite.rect.y);
                var width = Mathf.CeilToInt(currentSprite.rect.width);
                var height = Mathf.CeilToInt(currentSprite.rect.height);
                var currentBytes = new byte[width][][];

                var printing = false;

                if (i == 12 || i == 42)
                {
                    sb.AppendLine($"Printing sprite #{i}");
                    printing = true;
                }

                for (int x = 0; x < width; x++)
                {
                    currentBytes[x] = new byte[height][];
                    for (int y = 0; y < height; y++)
                    {
                        var color = texture.GetPixel(xOrigin + x, yOrigin + y);
                        var r = (byte)Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue);
                        var g = (byte)Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue);
                        var b = (byte)Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue);
                        var a = (byte)Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue);

                        if (printing)
                            sb.AppendLine($"({x},{y}) = {r},{g},{b},{a}");
                        currentBytes[x][y] = new byte[4] { r, g, b, a };
                    }
                }
                if (printing)
                {
                    File.WriteAllText($"C:\\ABC\\opt-{i}.txt", sb.ToString());
                    sb.Clear();
                }
                bytes[i] = currentBytes;
            }
        }

        return (bytes, sprites);
    }

    private (MyColor[][][] colors, byte[][][][] bytes, Sprite[] sprites) getColors(Sprite sprite)
    {
        var (bytes, sprites) = getBytes(sprite);
        var colors = new MyColor[bytes.Length][][];
        for (int i = 0; i < colors.Length; i++)
        {
            colors[i] = new MyColor[bytes[i].Length][];
            for (int x = 0; x < colors[i].Length; x++)
            {
                colors[i][x] = new MyColor[bytes[i][x].Length];
                for (int y = 0; y < colors[i][x].Length; y++)
                {
                    var currentBytes = bytes[i][x][y];
                    colors[i][x][y] = new MyColor(currentBytes[0], currentBytes[1], currentBytes[2], currentBytes[3]);
                }
            }
        }
        return (colors, bytes, sprites);

        //var texture = sprite.texture;
        //var path = AssetDatabase.GetAssetPath(sprite);
        //var allAssetsAtPath = AssetDatabase.LoadAllAssetsAtPath(path);
        //var allSptitesAtPath = allAssetsAtPath.OfType<Sprite>().ToArray();
        //var ti = AssetImporter.GetAtPath(path) as TextureImporter;
        //var fullPath = $"{Application.dataPath.Substring(0, Application.dataPath.Length - "Assets".Length)}{path}";
        ////Debug.LogError($"path = {fullPath}");
        //texture = new Texture2D(1, 1, TextureFormat.ARGB32, false);
        //texture.filterMode = FilterMode.Point;
        //texture.LoadImage(File.ReadAllBytes(fullPath));
        ////return null;
        //ti.isReadable = true;
        //ti.SaveAndReimport();

        //var spritesCount = ti.spritesheet.Length;
        //var colors = default(MyColor[][][]);
        //var sprites = default(Sprite[]);
        //var sb = new StringBuilder();
        //if (spritesCount == 0) //If there're no items in spritesheet - it means there is a single sprite in asset.
        //{
        //    colors = new MyColor[1][][];
        //    sprites = new Sprite[1];
        //    sprites[0] = sprite;

        //    var tex = sprite.texture;
        //    var currentColors = new MyColor[tex.width][];
        //    for (int x = 0; x < tex.width; x++)
        //    {
        //        currentColors[x] = new MyColor[tex.height];
        //        for (int y = 0; y < tex.height; y++)
        //        {
        //            var color = texture.GetPixel(x, y);
        //            currentColors[x][y] = new MyColor(
        //                Convert.ToByte(Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue)),
        //                Convert.ToByte(Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue)),
        //                Convert.ToByte(Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue)),
        //                Convert.ToByte(Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue))
        //            );
        //        }
        //    }
        //    colors[0] = currentColors;
        //}
        //else
        //{
        //    colors = new MyColor[spritesCount][][];
        //    sprites = new Sprite[spritesCount];

        //    for (int i = 0; i < spritesCount; i++)
        //    {
        //        var currentSprite = ti.spritesheet[i];
        //        sprites[i] = allSptitesAtPath.Where(s => s.name == currentSprite.name).First();

        //        var xOrigin = Mathf.FloorToInt(currentSprite.rect.x);
        //        var yOrigin = Mathf.CeilToInt(currentSprite.rect.y);
        //        var width = Mathf.CeilToInt(currentSprite.rect.width);
        //        var height = Mathf.CeilToInt(currentSprite.rect.height);
        //        var currentColors = new MyColor[width][];

        //        var printing = false;

        //        if (i == 12 || i == 42)
        //        {
        //            sb.AppendLine($"Printing sprite #{i}");
        //            printing = true;
        //        }

        //        for (int x = 0; x < width; x++)
        //        {
        //            currentColors[x] = new MyColor[height];
        //            for (int y = 0; y < height; y++)
        //            {
        //                var color = texture.GetPixel(xOrigin + x, yOrigin + y);
        //                var r = (byte)Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue);
        //                var g = (byte)Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue);
        //                var b = (byte)Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue);
        //                var a = (byte)Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue);
        //                if (printing)
        //                    sb.AppendLine($"({x},{y}) = {r},{g},{b},{a}");
        //                currentColors[x][y] = new MyColor(r, g, b, a);
        //            }
        //        }
        //        if (printing)
        //        {
        //            File.WriteAllText($"C:\\ABC\\opt-{i}.txt", sb.ToString());
        //            sb.Clear();
        //        }
        //        colors[i] = currentColors;
        //    }
        //}

        //return (colors, sprites);
    }
}
