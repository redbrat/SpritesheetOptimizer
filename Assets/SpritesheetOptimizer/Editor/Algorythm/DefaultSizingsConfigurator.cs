using System.Collections.Generic;

public class DefaultSizingsConfigurator : ISizingsConfigurator
{
    public virtual IEnumerable<MyVector2> ConfigureSizings(IEnumerable<MyVector2> result, int spritesCount, int xSize, int ySize)
    {
        var resultList = new List<MyVector2>();
        var x = xSize;
        var y = ySize;

        //Если у нас не квадрат - сначала доводим до квадрата
        if (x > y)
            while (x != y)
                resultList.Add(new MyVector2(x--, y));
        else if (y > x)
            while (x != y)
                resultList.Add(new MyVector2(x, y--));

        //Когда довели до квадрата - используем каждый доступный подквадрат
        return defaultAreaSizingsFunction(resultList, new MyVector2(x, y));
    }

    protected IEnumerable<MyVector2> defaultAreaSizingsFunction(List<MyVector2> resultList, MyVector2 area)
    {
        var currentArea = area;
        var counter = area.X != area.Y ? area.X < area.Y ? 1 : 2 : 0;
        while (currentArea.X > 0 && currentArea.Y > 0)
        {
            resultList.Add(currentArea);
            if (counter % 3 == 0)
                currentArea = new MyVector2(currentArea.X - 1, currentArea.Y);
            else if (counter % 3 == 1)
                currentArea = new MyVector2(currentArea.X + 1, currentArea.Y - 1);
            else
                currentArea = new MyVector2(currentArea.X - 1, currentArea.Y);

            counter++;
        }

        return resultList;
    }
}