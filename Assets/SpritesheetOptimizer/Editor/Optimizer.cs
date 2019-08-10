using System;
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

        if (_sprite != null && _cts == null && GUILayout.Button("Try"))
        {
            var algorithmBulder = new AlgorythmBuilder();
            var algorythm = algorithmBulder
                .AddSizingsConfigurator<PickySizingConfigurator>(_pickinessLevel)
                .AddScoreCounter<DefaultScoreCounter>()
                .SetAreaEnumerator<DefaultAreaEnumerator>()
                .SetAreasFreshmentSpan(_areaFreshmentSpan)
                .SetAreasVolatilityRange(_areasVolatilityRange)
                .Build(getColors(_sprite));
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
        ti.isReadable = true;
        ti.SaveAndReimport();

        var spritesCount = ti.spritesheet.Length;
        var sprites = (MyColor[][][])null;

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
        }

        return sprites;
    }
}
