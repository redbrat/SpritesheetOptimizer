using UnityEngine;

public abstract class FunctionBase4<T1, T2, T3, T4, TResult> : ScriptableObject
{
    public abstract TResult Invoke(T1 param1, T2 param2, T3 param3, T4 param4);
}
