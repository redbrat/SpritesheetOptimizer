using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

[Serializable]
public class MainWindow : EditorWindow
{
    private static MainWindow _intance;
    private Convergence _convergence;

    private static Sprite _selectedSprite;

    private static string _atlasFolderName = "Assets/Atlas";
    private static TextAsset _importedFromCuda;

    [MenuItem("Optimizer/Main Window")]
    private static void windowInitializer()
    {
        var convergenceName = default(StringValue);
        if (!AssetsUtility.FindAsset("ConvergenceName", out convergenceName))
            throw new ApplicationException("Can't find ConvergenceName asset! Try to reimport the asset from asset store.");
        var convergence = default(Convergence);
        if (!AssetsUtility.FindAsset(convergenceName.Value, out convergence))
            throw new ApplicationException("Can't find Convergence asset! Try to reimport the asset from asset store.");

        _intance = GetWindow<MainWindow>();
        _intance._convergence = convergence;
    }

    private void OnGUI()
    {
        _selectedSprite = EditorGUILayout.ObjectField("Sprite: ", _selectedSprite, typeof(Sprite), false) as Sprite;
        if (_selectedSprite != null && GUILayout.Button("Send to CUDA"))
            _convergence.ExportAllSprites(_selectedSprite);

        _atlasFolderName = EditorGUILayout.TextField("Atlas folder: ", _atlasFolderName);
        _importedFromCuda = EditorGUILayout.ObjectField("Imported from cuda file: ", _importedFromCuda, typeof(TextAsset), false) as TextAsset;
        if (_importedFromCuda != default && GUILayout.Button("Parse Cuda"))
        {
            _convergence.Import(_importedFromCuda.bytes, _importedFromCuda.name);
            //var bytes = _importedFromCuda.bytes;
            //Debug.LogError($"bytes length = {bytes.Length}");

            //var metaLength = BitConverter.ToInt32(bytes, 1);
            //Debug.LogError($"metaLength = {metaLength}");

            //var metaText = Encoding.UTF8.GetString(bytes, 5, metaLength);
            //Debug.Log($"metaText = {metaText}");

            //var meta = JsonUtility.FromJson<FormatMeta>(metaText);
            //var sprites = new Sprite[meta.SpriteInfos.Length];
            //for (int i = 0; i < sprites.Length; i++)
            //{
            //    var spriteAsset = (Sprite)AssetDatabase.LoadAllAssetsAtPath(meta.SpriteInfos[i].PathToFile).Where(a => a.name == meta.SpriteInfos[i].NameOfSprite).FirstOrDefault();
            //    if (spriteAsset == default)
            //        throw new ApplicationException($"Sprite asset wan't found at {meta.SpriteInfos[i]} -> {meta.SpriteInfos[i].NameOfSprite}");
            //    sprites[i] = spriteAsset;
            //}

            //var currentByteOffset = 5 + metaLength;
            //var atlasLength = BitConverter.ToInt32(bytes, currentByteOffset);
            //currentByteOffset += 4;
            //var spritesCount = BitConverter.ToInt32(bytes, currentByteOffset);
            //currentByteOffset += 4;
            //var maxChunksInSpriteCountLength = BitConverter.ToInt16(bytes, currentByteOffset);
            //currentByteOffset += 2;

            //var atlasChunkSpriteIndexLength = BitConverter.ToInt16(bytes, currentByteOffset);
            //currentByteOffset += 2;
            //var atlasXLength = BitConverter.ToInt16(bytes, currentByteOffset);
            //currentByteOffset += 2;
            //var atlasYLength = BitConverter.ToInt16(bytes, currentByteOffset);
            //currentByteOffset += 2;
            //var atlasWidthLength = BitConverter.ToInt16(bytes, currentByteOffset);
            //currentByteOffset += 2;
            //var atlasHeightLength = BitConverter.ToInt16(bytes, currentByteOffset);
            //currentByteOffset += 2;

            //var atlasLengthLength = BitConverter.ToInt16(bytes, currentByteOffset);
            //currentByteOffset += 4;
            //var chunkOffsetXLength = BitConverter.ToInt16(bytes, currentByteOffset);
            //currentByteOffset += 2;
            //var chunkOffsetYLength = BitConverter.ToInt16(bytes, currentByteOffset);
            //currentByteOffset += 2;

            //Debug.Log($"atlasLength = {atlasLength}");
            //Debug.Log($"spritesCount = {spritesCount}");
            //Debug.Log($"maxChunksInSpriteCountLength = {maxChunksInSpriteCountLength}");
            //Debug.Log($"atlasChunkSpriteIndexLength = {atlasChunkSpriteIndexLength}");
            //Debug.Log($"atlasXLength = {atlasXLength}");
            //Debug.Log($"atlasYLength = {atlasYLength}");
            //Debug.Log($"atlasWidthLength = {atlasWidthLength}");
            //Debug.Log($"atlasHeightLength = {atlasHeightLength}");
            //Debug.Log($"atlasLengthLength = {atlasLengthLength}");
            //Debug.Log($"chunkOffsetXLength = {chunkOffsetXLength}");
            //Debug.Log($"chunkOffsetYLength = {chunkOffsetYLength}");

            //var bitOffset = currentByteOffset * 8;

            //var atlasTextures = new Sprite[atlasLength];
            //for (int i = 0; i < atlasLength; i++)
            //{
            //    var index = readBits(bytes, bitOffset, atlasChunkSpriteIndexLength);
            //    bitOffset += atlasChunkSpriteIndexLength;
            //    var x = readBits(bytes, bitOffset, atlasXLength);
            //    bitOffset += atlasXLength;
            //    var y = readBits(bytes, bitOffset, atlasYLength);
            //    bitOffset += atlasYLength;
            //    var width = readBits(bytes, bitOffset, atlasWidthLength);
            //    bitOffset += atlasWidthLength;
            //    var height = readBits(bytes, bitOffset, atlasHeightLength);
            //    bitOffset += atlasHeightLength;
            //    Debug.Log($"{i}. index {index}, x {x}, y {y}, width {width}, height {height}");

            //    var atlasTexture = new Texture2D(width, height, TextureFormat.ARGB32, false, false);
            //    atlasTexture.filterMode = FilterMode.Point;

            //    var originSprite = sprites[index];
            //    for (int xx = 0; xx < width; xx++)
            //    {
            //        for (int yy = 0; yy < height; yy++)
            //        {
            //            var originX = Mathf.FloorToInt(originSprite.rect.xMin) + x + xx;
            //            var originY = Mathf.FloorToInt(originSprite.rect.yMin) + y + yy;

            //            atlasTexture.SetPixel(xx, yy, originSprite.texture.GetPixel(originX, originY));
            //        }
            //    }
            //    atlasTexture.Apply();

            //    var path = Path.Combine(_atlasFolderName, $"{i}.png");
            //    File.WriteAllBytes(path, atlasTexture.EncodeToPNG());
            //    AssetDatabase.ImportAsset(path);
            //    var ti = AssetImporter.GetAtPath(path) as TextureImporter;
            //    ti.textureType = TextureImporterType.Sprite;
            //    ti.filterMode = FilterMode.Point;
            //    ti.alphaIsTransparency = true;
            //    ti.mipmapEnabled = false;
            //    ti.spriteImportMode = SpriteImportMode.Single;
            //    ti.spritePivot = Vector2.down + Vector2.right;
            //    ti.isReadable = true;
            //    ti.crunchedCompression = false;
            //    ti.textureCompression = TextureImporterCompression.Uncompressed;

            //    var texSettings = new TextureImporterSettings();
            //    ti.ReadTextureSettings(texSettings);
            //    texSettings.spriteAlignment = (int)SpriteAlignment.BottomLeft;
            //    ti.SetTextureSettings(texSettings);

            //    AssetDatabase.ImportAsset(path);

            //    atlasTextures[i] = AssetDatabase.LoadAssetAtPath<Sprite>(path);
            //}

            //var optimizedSprites = new OptimizedSprite[spritesCount];
            //var chunkList = new List<ChunkStruct>();

            //var chunksOffset = bitOffset + maxChunksInSpriteCountLength * spritesCount;
            //for (int i = 0; i < spritesCount; i++)
            //{
            //    var chunksInSpriteCount = readBits(bytes, bitOffset, maxChunksInSpriteCountLength);
            //    bitOffset += maxChunksInSpriteCountLength;

            //    for (int c = 0; c < chunksInSpriteCount; c++)
            //    {
            //        var atlasIndex = readBits(bytes, chunksOffset, atlasLengthLength);
            //        chunksOffset += atlasLengthLength;
            //        var offsetX = readBits(bytes, chunksOffset, chunkOffsetXLength);
            //        chunksOffset += chunkOffsetXLength;
            //        var offsetY = readBits(bytes, chunksOffset, chunkOffsetYLength);
            //        chunksOffset += chunkOffsetYLength;

            //        chunkList.Add(new ChunkStruct(atlasIndex, offsetX, offsetY));
            //    }
            //    var newOptimizedSprite = (OptimizedSprite)CreateInstance(typeof(OptimizedSprite));
            //    newOptimizedSprite.Chunks = chunkList.ToArray();
            //    //AssetDatabase.CreateAsset(newOptimizedSprite, );
            //    chunkList.Clear();
            //    optimizedSprites[i] = newOptimizedSprite;
            //}
        }
    }

    //private int readBits(byte[] bytes, int bitOffset, int valueLength)
    //{
    //    var byteIndex = bitOffset / 8;
    //    var bitIndex = bitOffset % 8;
    //    var availableSpace = 32 - bitIndex;
    //    if (availableSpace >= valueLength)
    //        return (BitConverter.ToInt32(bytes, byteIndex) >> bitIndex) & valueLength.GetMaxInt();
    //    else
    //    {
    //        var part1 = (BitConverter.ToInt32(bytes, byteIndex) >> bitIndex) & availableSpace.GetMaxInt();
    //        var part2 = (BitConverter.ToInt32(bytes, byteIndex + 4) & (valueLength - availableSpace).GetMaxInt()) << availableSpace;
    //        return part1 | part2;
    //    }
    //}
}
