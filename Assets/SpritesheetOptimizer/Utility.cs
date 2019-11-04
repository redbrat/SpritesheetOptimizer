using System.Reflection;

public static class Utility
{
    public static T MemberwiseClone<T>(this T original)
    {
        var type = typeof(T);
        var methodInfo = type.GetMethod("MemberwiseClone", BindingFlags.NonPublic | BindingFlags.Instance);
        return (T)methodInfo.Invoke(original, new object[0]);
    }
}
