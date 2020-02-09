using UnityEngine;

public abstract class MethodBase2<T1, T2> : ScriptableObject
{
    public abstract void Invoke(T1 param1, T2 param2);
}
