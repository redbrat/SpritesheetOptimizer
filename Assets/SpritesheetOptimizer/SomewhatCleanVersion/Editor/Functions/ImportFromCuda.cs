using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

[CreateAssetMenu(fileName = nameof(ImportFromCuda), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.FunctionsName + "/" + nameof(ImportFromCuda), order = 0)]
public class ImportFromCuda : FunctionBase2<byte[], DefaultAsset, Import>
{
    [SerializeField]
    private StringValue _atlasName;
    [SerializeField]
    private StringValue _oprtimizedSpritesName;

    [SerializeField]
    private ReadBits _readBits;
    [SerializeField]
    private WriteAndImportSprite _writeAndImportSprite;

    public override Import Invoke(byte[] bytes, DefaultAsset parent)
    {
        var metaLength = BitConverter.ToInt32(bytes, 1);

        var metaText = Encoding.UTF8.GetString(bytes, 5, metaLength);

        var meta = JsonUtility.FromJson<FormatMeta>(metaText);
        var sprites = new Sprite[meta.SpriteInfos.Length];
        for (int i = 0; i < sprites.Length; i++)
        {
            var spriteAsset = (Sprite)AssetDatabase.LoadAllAssetsAtPath(meta.SpriteInfos[i].PathToFile).Where(a => a.name == meta.SpriteInfos[i].NameOfSprite).FirstOrDefault();
            if (spriteAsset == default)
                throw new ApplicationException($"Sprite asset wan't found at {meta.SpriteInfos[i]} -> {meta.SpriteInfos[i].NameOfSprite}");
            sprites[i] = spriteAsset;
        }

        var currentByteOffset = 5 + metaLength;
        var atlasLength = BitConverter.ToInt32(bytes, currentByteOffset);
        currentByteOffset += 4;
        var spritesCount = BitConverter.ToInt32(bytes, currentByteOffset);
        currentByteOffset += 4;
        var maxChunksInSpriteCountLength = BitConverter.ToInt16(bytes, currentByteOffset);
        currentByteOffset += 2;

        var atlasChunkSpriteIndexLength = BitConverter.ToInt16(bytes, currentByteOffset);
        currentByteOffset += 2;
        var atlasXLength = BitConverter.ToInt16(bytes, currentByteOffset);
        currentByteOffset += 2;
        var atlasYLength = BitConverter.ToInt16(bytes, currentByteOffset);
        currentByteOffset += 2;
        var atlasWidthLength = BitConverter.ToInt16(bytes, currentByteOffset);
        currentByteOffset += 2;
        var atlasHeightLength = BitConverter.ToInt16(bytes, currentByteOffset);
        currentByteOffset += 2;

        var atlasLengthLength = BitConverter.ToInt16(bytes, currentByteOffset);
        currentByteOffset += 4;
        var chunkOffsetXLength = BitConverter.ToInt16(bytes, currentByteOffset);
        currentByteOffset += 2;
        var chunkOffsetYLength = BitConverter.ToInt16(bytes, currentByteOffset);
        currentByteOffset += 2;

        var bitOffset = currentByteOffset * 8;

        var atlasTexturesFolder = parent.GetOfCreateSubFolder(_atlasName.Value);
        var atlasTextures = new Sprite[atlasLength];
        for (int i = 0; i < atlasLength; i++)
        {
            var index = _readBits.Invoke(bytes, bitOffset, atlasChunkSpriteIndexLength);
            bitOffset += atlasChunkSpriteIndexLength;
            var x = _readBits.Invoke(bytes, bitOffset, atlasXLength);
            bitOffset += atlasXLength;
            var y = _readBits.Invoke(bytes, bitOffset, atlasYLength);
            bitOffset += atlasYLength;
            var width = _readBits.Invoke(bytes, bitOffset, atlasWidthLength);
            bitOffset += atlasWidthLength;
            var height = _readBits.Invoke(bytes, bitOffset, atlasHeightLength);
            bitOffset += atlasHeightLength;
            Debug.Log($"{i}. index {index}, x {x}, y {y}, width {width}, height {height}");

            var atlasTexture = new Texture2D(width, height, TextureFormat.ARGB32, false, false);
            atlasTexture.filterMode = FilterMode.Point;

            var originSprite = sprites[index];
            for (int xx = 0; xx < width; xx++)
            {
                for (int yy = 0; yy < height; yy++)
                {
                    var originX = Mathf.FloorToInt(originSprite.rect.xMin) + x + xx;
                    var originY = Mathf.FloorToInt(originSprite.rect.yMin) + y + yy;

                    atlasTexture.SetPixel(xx, yy, originSprite.texture.GetPixel(originX, originY));
                }
            }
            atlasTexture.Apply();

            atlasTextures[i] = _writeAndImportSprite.Invoke(atlasTexture.EncodeToPNG(), atlasTexturesFolder, $"{i}.png");
            DestroyImmediate(atlasTexture);
        }

        var optimizedSprites = new OptimizedSprite[spritesCount];
        var chunkList = new List<OptimizedSpriteChunk>();
        var optimizedSpritesFolder = parent.GetOfCreateSubFolder(_oprtimizedSpritesName.Value);
        var chunksOffset = bitOffset + maxChunksInSpriteCountLength * spritesCount;
        for (int i = 0; i < spritesCount; i++)
        {
            var chunksInSpriteCount = _readBits.Invoke(bytes, bitOffset, maxChunksInSpriteCountLength);
            bitOffset += maxChunksInSpriteCountLength;

            for (int c = 0; c < chunksInSpriteCount; c++)
            {
                var atlasIndex = _readBits.Invoke(bytes, chunksOffset, atlasLengthLength);
                chunksOffset += atlasLengthLength;
                var offsetX = _readBits.Invoke(bytes, chunksOffset, chunkOffsetXLength);
                chunksOffset += chunkOffsetXLength;
                var offsetY = _readBits.Invoke(bytes, chunksOffset, chunkOffsetYLength);
                chunksOffset += chunkOffsetYLength;

                var newChunkStruct = new OptimizedSpriteChunk();
                var chunkSprite = atlasTextures[atlasIndex];
                newChunkStruct.Sprite = chunkSprite;
                newChunkStruct.X = offsetX;
                newChunkStruct.Y = offsetY;
                chunkList.Add(newChunkStruct);
            }
            var newOptimizedSprite = (OptimizedSprite)CreateInstance(typeof(OptimizedSprite));
            newOptimizedSprite.OriginalSprite = sprites[i];
            newOptimizedSprite.Pivot = meta.SpriteInfos[i].Pivot;
            newOptimizedSprite.Width = meta.SpriteInfos[i].Size.x;
            newOptimizedSprite.Height = meta.SpriteInfos[i].Size.y;
            newOptimizedSprite.Chunks = chunkList.ToArray();
            newOptimizedSprite.CreateAsAsset(optimizedSpritesFolder.GetPathToFileName($"{i}.asset"));
            //newOptimizedSprite.DebugSprite();
            optimizedSprites[i] = newOptimizedSprite;
            chunkList.Clear();
        }

        var import = CreateInstance<Import>();
        import.Chunks = atlasTextures;
        import.Folder = parent;
        import.Meta = meta;
        import.OptimizedSprites = optimizedSprites;
        import.CreateAsAsset(parent.GetPathToFileName("import.asset"));
        return import;
    }
}
