using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using UnityEditor;
using UnityEngine;

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

    [MenuItem("Optimizer/Optimize")]
    private static void Main()
    {
        _intance = GetWindow<Optimizer>();
    }

    private ProgressReport _operationProgressReport;
    private ProgressReport _overallProgressReport;
    private CancellationTokenSource _cts;

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

        if (_sprite != null && _cts == null && GUILayout.Button("Try"))
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

        Repaint();
    }

    private async void launch(Algorythm algorythm, Sprite[] sprites, MyColor[][][] colors, MyVector2Float[] pivots)
    {
        await algorythm.Initialize(_resolution, _cts.Token);
        var correlations = await algorythm.Run();
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

        saveSpritesInfo(areasPerSprite, sprites);
    }

    private void saveSpritesInfo(Dictionary<int, List<SpriteChunk>> areasPerSprite, Sprite[] sprites)
    {
        var newSpritesInfo = ScriptableObject.CreateInstance<UnityOptimizedSpritesStructure>();
        AssetDatabase.CreateAsset(newSpritesInfo, _resultFileName);
        AssetDatabase.CreateFolder(Path.GetDirectoryName(_resultFileName), Path.GetFileNameWithoutExtension(_resultFileName));
        var spritesDirectory = Path.Combine(Path.GetDirectoryName(_resultFileName), Path.GetFileNameWithoutExtension(_resultFileName));

        newSpritesInfo.Sprites = sprites;

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
                var pivot = pivots[info.SpriteIndex];
                var width = colors[info.SpriteIndex].Length;
                var height = colors[info.SpriteIndex][0].Length;
                var ppu = pixelPerUnits[info.SpriteIndex];
                var offsetX = Mathf.FloorToInt(pivot.X * width / ppu);
                var offsetY = Mathf.FloorToInt(pivot.Y * height / ppu);
                var pivotedInfo = new MyAreaCoordinates(info.SpriteIndex, info.X - offsetX, info.Y - offsetY, info.Width, info.Height);
                if (!result.ContainsKey(info.SpriteIndex))
                    result.Add(info.SpriteIndex, new List<SpriteChunk>());

                result[info.SpriteIndex].Add(new SpriteChunk(correlations[i].Colors, pivotedInfo));
            }
        }

        return result;
    }

    private (MyColor[][][] colors, Sprite[] sprites) getColors(Sprite sprite)
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
        var colors = default(MyColor[][][]);
        var sprites = default(Sprite[]);
        var sb = new StringBuilder();
        if (spritesCount == 0) //If there're no items in spritesheet - it means there is a single sprite in asset.
        {
            colors = new MyColor[1][][];
            sprites = new Sprite[1];
            sprites[0] = sprite;

            var tex = sprite.texture;
            var currentColors = new MyColor[tex.width][];
            for (int x = 0; x < tex.width; x++)
            {
                currentColors[x] = new MyColor[tex.height];
                for (int y = 0; y < tex.height; y++)
                {
                    var color = texture.GetPixel(x, y);
                    currentColors[x][y] = new MyColor(
                        Convert.ToByte(Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue)),
                        Convert.ToByte(Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue)),
                        Convert.ToByte(Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue)),
                        Convert.ToByte(Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue))
                    );
                }
            }
            colors[0] = currentColors;
        }
        else
        {
            colors = new MyColor[spritesCount][][];
            sprites = new Sprite[spritesCount];

            for (int i = 0; i < spritesCount; i++)
            {
                var currentSprite = ti.spritesheet[i];
                sprites[i] = allSptitesAtPath.Where(s => s.name == currentSprite.name).First();

                var xOrigin = Mathf.FloorToInt(currentSprite.rect.x);
                var yOrigin = Mathf.CeilToInt(currentSprite.rect.y);
                var width = Mathf.CeilToInt(currentSprite.rect.width);
                var height = Mathf.CeilToInt(currentSprite.rect.height);
                var currentColors = new MyColor[width][];

                var printing = false;

                if (i == 12 || i == 42)
                {
                    sb.AppendLine($"Printing sprite #{i}");
                    printing = true;
                }

                for (int x = 0; x < width; x++)
                {
                    currentColors[x] = new MyColor[height];
                    for (int y = 0; y < height; y++)
                    {
                        var color = texture.GetPixel(xOrigin + x, yOrigin + y);
                        var r = (byte)Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue);
                        var g = (byte)Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue);
                        var b = (byte)Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue);
                        var a = (byte)Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue);
                        if (printing)
                            sb.AppendLine($"({x},{y}) = {r},{g},{b},{a}");
                        currentColors[x][y] = new MyColor(r, g, b, a);
                    }
                }
                if (printing)
                {
                    File.WriteAllText($"C:\\ABC\\opt-{i}.txt", sb.ToString());
                    sb.Clear();
                }
                colors[i] = currentColors;
            }
        }

        return (colors, sprites);
    }
}
