using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

public class PickySizingConfigurator : DefaultSizingsConfigurator
{
    [Serializable]
    public enum PickynessLevel
    {
        None = 0,
        Low, //убираем каждый второй не-квадрат
        Medium, //убираем все неквадраты
        High //Убираем все неквадраты и каждый второй квадрат
    }

    private readonly PickynessLevel _pickyness;

    public PickySizingConfigurator(PickynessLevel pickiness)
    {
        _pickyness = pickiness;
    }

    public override IEnumerable<MyVector2> ConfigureSizings(IEnumerable<MyVector2> result, int spritesCount, int xSize, int ySize, CancellationToken ct)
    {
        var resultList = base.ConfigureSizings(result, spritesCount, xSize, ySize, ct).ToList();
        var counter = 0;
        switch (_pickyness)
        {
            case PickynessLevel.Low:
                for (int i = 0; i < resultList.Count; i++)
                {
                    if (i == 0 || i == resultList.Count - 1)
                        continue;
                    var area = resultList[i];
                    if (area.X == area.Y)
                        continue;
                    if (((counter ^ (counter >> 1)) & 1) == 0)
                    {
                        resultList.RemoveAt(i);
                        i--;
                    }
                    counter = (counter + 1) % 4;
                }
                break;
            case PickynessLevel.Medium:
                for (int i = 0; i < resultList.Count; i++)
                {
                    if (i == 0 || i == resultList.Count - 1)
                        continue;
                    var area = resultList[i];
                    if (area.X == area.Y)
                        continue;
                    resultList.RemoveAt(i);
                    i--;
                }
                break;
            case PickynessLevel.High:
                for (int i = 0; i < resultList.Count; i++)
                {
                    if (i == 0 || i == resultList.Count - 1)
                        continue;
                    var area = resultList[i];
                    if (area.X == area.Y)
                    {
                        if (counter++ % 2 == 0)
                        {
                            resultList.RemoveAt(i);
                            i--;
                        }
                        continue;
                    }
                    resultList.RemoveAt(i);
                    i--;
                }
                break;
            default:
                break;
        }

        return resultList;
    }
}
