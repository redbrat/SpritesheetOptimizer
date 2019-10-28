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
    private static string _resultFileName = "Assets/spriteChunks.asset";

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
        _computeMode = (ComputeMode)EditorGUILayout.EnumPopup($"Compute on", _computeMode);

        if (_sprite != null && _cts == null && GUILayout.Button("Try"))
        {
            var colorResults = getColors(_sprite);
            var algorithmBulder = new AlgorythmBuilder();
            var algorythm = algorithmBulder
                .AddSizingsConfigurator<PickySizingConfigurator>(_pickinessLevel)
                .AddScoreCounter<DefaultScoreCounter>()
                .SetAreaEnumerator<DefaultAreaEnumerator>()
                .SetAreasFreshmentSpan(_areaFreshmentSpan)
                .SetAreasVolatilityRange(_areasVolatilityRange)
                .Build(colorResults.colors, _computeMode);
            _operationProgressReport = algorythm.OperationProgressReport;
            _overallProgressReport = algorythm.OverallProgressReport;
            _cts = new CancellationTokenSource();
            launch(algorythm, colorResults.sprites);
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

    private async void launch(Algorythm algorythm, Sprite[] sprites)
    {
        await algorythm.Initialize(_resolution, _cts.Token);
        var correlations = await algorythm.Run();
        var areasPerSprite = getAreasPerSprite(correlations);

        Debug.Log($"Максимальное кол-во областей в одном спрайте: {areasPerSprite.OrderByDescending(aps => aps.Value.Count).First().Value.Count}");
        Debug.Log($"Минимальное кол-во областей в одном спрайте: {areasPerSprite.OrderBy(aps => aps.Value.Count).First().Value.Count}");
        Debug.Log($"Общее кол-во непрозрачных пикселей: {correlations.Aggregate(0, (count, cor) => count += cor.Colors.Length, count => count)}");
        Debug.Log($"Общее кол-во ссылок: {correlations.Aggregate(0, (count, cor) => count += cor.Coordinates.Length, count => count)}");

        _operationProgressReport = null;
        _cts = null;

        saveSpritesInfo(areasPerSprite, sprites);
    }

    private void saveSpritesInfo(Dictionary<int, List<SpriteChunk>> areasPerSprite, Sprite[] sprites)
    {
        var newSpritesInfo = ScriptableObject.CreateInstance<UnityOptimizedSpritesStructure>();
        AssetDatabase.CreateAsset(newSpritesInfo, _resultFileName);

        newSpritesInfo.Sprites = sprites;
        
        var chunks = new SpriteChunkArrayWrapper[sprites.Length];
        foreach (var kvp in areasPerSprite)
        {
            var spriteIndex = kvp.Key;
            var currentChunks = kvp.Value;
            var currentChunksArray = currentChunks.ToArray();

            var references = new Dictionary<MySerializableColor[][], ColorsReference>();

            for (int i = 0; i < currentChunksArray.Length; i++)
            {
                var currentColors = currentChunksArray[i].Colors;
                if (!references.ContainsKey(currentColors))
                {
                    var newColorsReference = ScriptableObject.CreateInstance<ColorsReference>();
                    newColorsReference.Colors = currentColors;
                    references.Add(currentColors, newColorsReference);

                    AssetDatabase.AddObjectToAsset(newColorsReference, _resultFileName);
                }
                currentChunksArray[i].ColorsReference = references[currentColors];
                currentChunksArray[i].Colors = default;
            }

            chunks[spriteIndex] = new SpriteChunkArrayWrapper(currentChunksArray);
        }

        newSpritesInfo.Chunks = chunks;

        AssetDatabase.SaveAssets();
    }

    private Dictionary<int, List<SpriteChunk>> getAreasPerSprite(Algorythm.Correlation[] correlations)
    {
        var result = new Dictionary<int, List<SpriteChunk>>();

        for (int i = 0; i < correlations.Length; i++)
        {
            for (int j = 0; j < correlations[i].Coordinates.Length; j++)
            {
                var info = correlations[i].Coordinates[j];
                if (!result.ContainsKey(info.SpriteIndex))
                    result.Add(info.SpriteIndex, new List<SpriteChunk>());

                result[info.SpriteIndex].Add(new SpriteChunk(correlations[i].Colors, info));
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
        Debug.LogError($"path = {fullPath}");
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
