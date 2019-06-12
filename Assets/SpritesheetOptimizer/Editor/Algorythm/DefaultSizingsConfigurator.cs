using System.Collections.Generic;

public class DefaultSizingsConfigurator : ISizingsConfigurator
{
    public virtual IEnumerable<MyVector2> ConfigureSizings(IEnumerable<MyVector2> result, int spritesCount, int xSize, int ySize)
    {
        var minSize = xSize > ySize ? ySize : xSize;
        var maxSquare = new MyVector2(minSize, minSize);
        return defaultAreaSizingsFunction(maxSquare);
    }

    protected IEnumerable<MyVector2> defaultAreaSizingsFunction(MyVector2 area)
    {
        var resultList = new List<MyVector2>();

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