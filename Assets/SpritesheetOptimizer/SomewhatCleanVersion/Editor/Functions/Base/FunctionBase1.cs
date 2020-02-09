using UnityEngine;

public abstract class FunctionBase1<T, TResult> : ScriptableObject
{
    public abstract TResult Invoke(T param);
}
