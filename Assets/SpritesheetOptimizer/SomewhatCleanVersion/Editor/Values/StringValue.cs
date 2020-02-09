using UnityEngine;

[CreateAssetMenu(fileName = nameof(StringValue), menuName = CreateAssetMenuPaths.ProductName + "/" + CreateAssetMenuPaths.ValuesName + "/" + nameof(StringValue), order = 0)]
public class StringValue : ValueBase<string> { }
