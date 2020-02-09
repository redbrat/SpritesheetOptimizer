using UnityEngine;

public abstract class FunctionBase2<T1, T2, TResult> : ScriptableObject
{
    public abstract TResult Invoke(T1 param1, T2 param2);
}
