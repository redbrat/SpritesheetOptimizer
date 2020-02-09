using UnityEditor;
using UnityEngine;

[CreateAssetMenu(fileName = nameof(Convergence), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.RemoveLaterName + "/" + nameof(Convergence), order = 0)]
public class Convergence : ScriptableObject
{
    [SerializeField]
    private DefaultAsset _importsFolder;
    [SerializeField]
    private DefaultAsset _cudaDbFolder;

    [SerializeField]
    private GetSpritesBySprite _getSpritesBySprite;
    [SerializeField]
    private GetSpritesInfoBySprites _getSpritesInfoBySprites;
    [SerializeField]
    private GetCudaExport _getCudaExport;
    [SerializeField]
    private WriteAndImportBytes _writeAndImportBytes;
    [SerializeField]
    private GetPathByFolderAndName _getPathByFolderAndName;
    [SerializeField]
    private GetHashOfByteArray _getHashOfByteArray;

    [SerializeField]
    private ImportFromCuda _importFromCuda;

    public void ExportAllSprites(Sprite sprite)
    {
        var bytes = _getCudaExport.Invoke(_getSpritesInfoBySprites.Invoke(_getSpritesBySprite.Invoke(sprite)));
        _writeAndImportBytes.Invoke(bytes, _getPathByFolderAndName.Invoke(_cudaDbFolder, $"{_getHashOfByteArray.Invoke(bytes)}.bytes"));
    }

    public void Import(byte[] bytes, string folderName)
    {
        var parentFolder = _importsFolder.GetOfCreateSubFolder(folderName);
        _importFromCuda.Invoke(bytes, parentFolder);
    }
}
