using System;
using System.Collections.Generic;

public static class JaggedToMD
{
    private static readonly Type _arrayType = typeof(Array);

    public static Array ConvertToMultiDimentional(this Array jagged, bool checkJagginess = true)
    {
        var elementType = default(Type);
        var shape = default(int[]);
        if (!isJagged(jagged, out elementType, out shape))
            return jagged; //массив не jagged изначально - возвращаем его
        if (checkJagginess)
            checkJagginessRecursively(jagged, shape, new int?[shape.Length - 1], 0); //Проверяем, чтобы все подмассивы были одинаковой размерности
        return setValuesRecursively(0, shape, jagged, Array.CreateInstance(elementType, shape));
    }

    private static void checkJagginessRecursively(Array jagged, int[] shape, int?[] childLengths, int currentLevel)
    {
        var currentLength = shape[currentLevel];
        var currentChildLength = childLengths[currentLevel];
        for (int i = 0; i < currentLength; i++)
        {
            var currentElementArray = (Array)jagged.GetValue(i);
            if (!currentChildLength.HasValue)
                currentChildLength = currentElementArray.Length;
            else if (currentElementArray.Length != currentChildLength.Value)
                throw new ArgumentException($"Jagged array has uneven elements count in level {currentLength}");
            if (currentLevel + 2 < shape.Length) //last element isn't array
                checkJagginessRecursively(currentElementArray, shape, childLengths, currentLevel + 1);
        }
    }

    private static Array setValuesRecursively(int currentLevel, int[] shape, Array jagged, Array md)
    {
        var currentLevelLength = shape[currentLevel];
        var elementType = jagged.GetType().GetElementType();
        var isArray = _arrayType.IsAssignableFrom(elementType);

        for (int i = 0; i < currentLevelLength; i++)
        {
            if (isArray)
                setValuesRecursively(currentLevel + 1, shape, (Array)jagged.GetValue(i), md);
            else
                md.SetValue(jagged.GetValue(i), i);
        }

        return md;
    }

    private static bool isJagged(Array jagged, out Type elementType, out int[] shape)
    {
        elementType = default;

        var intList = new List<int>();
        var currentLevelArray = jagged;
        var currentElementType = jagged.GetType().GetElementType();
        while (elementType == default)
        {
            if (_arrayType.IsAssignableFrom(currentElementType))
            {
                intList.Add(currentLevelArray.Length);
                currentLevelArray = (Array)currentLevelArray.GetValue(0);
                currentElementType = currentLevelArray.GetType().GetElementType();
            }
            else if (intList.Count == 0) //Не jagged массив
            {
                elementType = currentElementType;
                shape = new int[] { jagged.Length };
                return false;
            }
            else
            {
                intList.Add(currentLevelArray.Length);
                elementType = currentElementType;
            }
        }

        shape = intList.ToArray();
        intList.Clear();
        return true;
    }
}
