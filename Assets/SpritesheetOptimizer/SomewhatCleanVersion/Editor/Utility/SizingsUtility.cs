using System.Collections.Generic;

public static class SizingsUtility
{
    private static List<MyVector2> _sizingsList = new List<MyVector2>();

    public static MyVector2[] GetStandardSizings(this MyVector2 area)
    {
        var x = area.X;
        var y = area.Y;

        //Если у нас не квадрат - сначала доводим до квадрата
        if (x > y)
            while (x != y)
                _sizingsList.Add(new MyVector2(x--, y));
        else if (y > x)
            while (x != y)
                _sizingsList.Add(new MyVector2(x, y--));

        //Когда довели до квадрата - используем каждый доступный подквадрат
        return x.GetStandardSizingsOfASquare();
    }

    public static MyVector2[] GetStandardSizingsOfASquare(this int side)
    {
        var currentX = side;
        var currentY = side;
        var counter = currentX != currentY ? currentX < currentY ? 1 : 2 : 0;
        while (currentX > 0 && currentY > 0)
        {
            _sizingsList.Add(new MyVector2(currentX, currentY));
            if (counter % 3 == 0)
                currentX--;
            else if (counter % 3 == 1)
            {
                currentX++;
                currentY--;
            }
            else
                currentX--;

            counter++;
        }

        var result = _sizingsList.ToArray();
        _sizingsList.Clear();
        return result;
    }
}
