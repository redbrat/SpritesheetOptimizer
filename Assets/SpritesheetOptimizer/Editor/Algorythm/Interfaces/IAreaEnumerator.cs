using System;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// Отвечает за перебор значений в спрайтах. Его фукнция - оптимизация.
/// </summary>
public interface IAreaEnumerator
{
    void EnumerateThroughSprite(MyVector2 areaSizing, int spriteIndex, Action<MyColor[][], int, int, int> action);
    void Enumerate(MyVector2 areaSizing, Action<MyColor[][], int, int, int> action);
    Task EnumerateParallel(MyVector2 areaSizing, Action<MyColor[][], int, int, int> action, CancellationToken ct);
    void EnumerateCopy(MyVector2 areaDimensions, Action<MyColor[][], int, int, int> action);
}