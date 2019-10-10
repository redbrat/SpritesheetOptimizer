using System;
using System.IO;
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
        _computeMode = (ComputeMode)EditorGUILayout.EnumPopup($"Compute on", _computeMode);

        if (_sprite != null && _cts == null && GUILayout.Button("Try"))
        {
            var algorithmBulder = new AlgorythmBuilder();
            var algorythm = algorithmBulder
                .AddSizingsConfigurator<PickySizingConfigurator>(_pickinessLevel)
                .AddScoreCounter<DefaultScoreCounter>()
                .SetAreaEnumerator<DefaultAreaEnumerator>()
                .SetAreasFreshmentSpan(_areaFreshmentSpan)
                .SetAreasVolatilityRange(_areasVolatilityRange)
                .Build(getColors(_sprite), _computeMode);
            _operationProgressReport = algorythm.OperationProgressReport;
            _overallProgressReport = algorythm.OverallProgressReport;
            _cts = new CancellationTokenSource();
            launch(algorythm);
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

    private async void launch(Algorythm algorythm)
    {
        await algorythm.Initialize(_resolution, _cts.Token);
        await algorythm.Run();
        _operationProgressReport = null;
        _cts = null;
    }

    private MyColor[][][] getColors(Sprite sprite)
    {
        var texture = sprite.texture;
        var path = AssetDatabase.GetAssetPath(sprite);
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
        var sprites = (MyColor[][][])null;
        var sb = new StringBuilder();
        if (spritesCount == 0) //If there're no items in spritesheet - it means there is a single sprite in asset.
        {
            sprites = new MyColor[1][][];

            var tex = sprite.texture;
            var colors = new MyColor[tex.width][];
            for (int x = 0; x < tex.width; x++)
            {
                colors[x] = new MyColor[tex.height];
                for (int y = 0; y < tex.height; y++)
                {
                    var color = texture.GetPixel(x, y);
                    colors[x][y] = new MyColor(
                        Convert.ToByte(Mathf.Clamp(color.r * byte.MaxValue, 0, byte.MaxValue)),
                        Convert.ToByte(Mathf.Clamp(color.g * byte.MaxValue, 0, byte.MaxValue)),
                        Convert.ToByte(Mathf.Clamp(color.b * byte.MaxValue, 0, byte.MaxValue)),
                        Convert.ToByte(Mathf.Clamp(color.a * byte.MaxValue, 0, byte.MaxValue))
                    );
                }
            }
            sprites[0] = colors;
        }
        else
        {
            sprites = new MyColor[spritesCount][][];

            for (int i = 0; i < spritesCount; i++)
            {
                var currentSprite = ti.spritesheet[i];

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
                sprites[i] = currentColors;
            }
        }

        return sprites;
    }
}
